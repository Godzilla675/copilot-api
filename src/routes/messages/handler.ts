import type { Context } from "hono"

import consola from "consola"
import { streamSSE } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import {
  createChatCompletions,
  type ChatCompletionChunk,
  type ChatCompletionResponse,
} from "~/services/copilot/create-chat-completions"
import {
  createResponses,
  type ResponsesApiResponse,
} from "~/services/copilot/create-responses"

import {
  type AnthropicMessagesPayload,
  type AnthropicStreamState,
} from "./anthropic-types"
import {
  translateToAnthropic,
  translateToOpenAI,
} from "./non-stream-translation"
import {
  createResponsesStreamState,
  translateResponsesStreamEvent,
} from "./responses-stream-translation"
import {
  translateAnthropicToResponses,
  translateResponsesToAnthropic,
} from "./responses-translation"
import { translateChunkToAnthropicEvents } from "./stream-translation"

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  const anthropicPayload = await c.req.json<AnthropicMessagesPayload>()
  consola.debug("Anthropic request payload:", JSON.stringify(anthropicPayload))

  const openAIPayload = translateToOpenAI(anthropicPayload)
  consola.debug(
    "Translated OpenAI request payload:",
    JSON.stringify(openAIPayload),
  )

  if (state.manualApprove) {
    await awaitApproval()
  }

  if (shouldUseResponsesApi(anthropicPayload)) {
    const responsesPayload = translateAnthropicToResponses(anthropicPayload)
    consola.debug(
      "Translated Responses payload:",
      JSON.stringify(responsesPayload),
    )

    const response = await createResponses(responsesPayload)

    if (isNonStreamingResponse(response)) {
      consola.debug(
        "Non-streaming response from Responses API:",
        JSON.stringify(response).slice(-400),
      )
      const anthropicResponse = translateResponsesToAnthropic(response)
      consola.debug(
        "Translated Anthropic response:",
        JSON.stringify(anthropicResponse),
      )
      return c.json(anthropicResponse)
    }

    consola.debug("Streaming response from Responses API")
    return streamSSE(c, async (stream) => {
      const streamState = createResponsesStreamState()

      for await (const rawEvent of response) {
        consola.debug("Responses raw stream event:", JSON.stringify(rawEvent))
        if (!rawEvent.data || rawEvent.data === "[DONE]") {
          continue
        }

        const events = translateResponsesStreamEvent(
          JSON.parse(rawEvent.data) as {
            type?: string
            [key: string]: unknown
          },
          streamState,
        )

        for (const event of events) {
          consola.debug("Translated Anthropic event:", JSON.stringify(event))
          await stream.writeSSE({
            event: event.type,
            data: JSON.stringify(event),
          })
        }
      }
    })
  }

  const response = await createChatCompletions(openAIPayload)

  if (isNonStreaming(response)) {
    consola.debug(
      "Non-streaming response from Copilot:",
      JSON.stringify(response).slice(-400),
    )
    const anthropicResponse = translateToAnthropic(response)
    consola.debug(
      "Translated Anthropic response:",
      JSON.stringify(anthropicResponse),
    )
    return c.json(anthropicResponse)
  }

  consola.debug("Streaming response from Copilot")
  return streamSSE(c, async (stream) => {
    const streamState: AnthropicStreamState = {
      messageStartSent: false,
      contentBlockIndex: 0,
      contentBlockOpen: false,
      toolCalls: {},
    }

    for await (const rawEvent of response) {
      consola.debug("Copilot raw stream event:", JSON.stringify(rawEvent))
      if (rawEvent.data === "[DONE]") {
        break
      }

      if (!rawEvent.data) {
        continue
      }

      const chunk = JSON.parse(rawEvent.data) as ChatCompletionChunk
      const events = translateChunkToAnthropicEvents(chunk, streamState)

      for (const event of events) {
        consola.debug("Translated Anthropic event:", JSON.stringify(event))
        await stream.writeSSE({
          event: event.type,
          data: JSON.stringify(event),
        })
      }
    }
  })
}

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")

const isNonStreamingResponse = (
  response: Awaited<ReturnType<typeof createResponses>>,
): response is ResponsesApiResponse => !(Symbol.asyncIterator in response)

function shouldUseResponsesApi(payload: AnthropicMessagesPayload): boolean {
  return (
    Boolean(payload.thinking)
    || payload.messages.some(
      (message) =>
        message.role === "assistant"
        && Array.isArray(message.content)
        && message.content.some((block) => block.type === "thinking"),
    )
  )
}
