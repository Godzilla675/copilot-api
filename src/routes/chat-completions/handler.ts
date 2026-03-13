import type { Context } from "hono"

import consola from "consola"
import { streamSSE, type SSEMessage } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { isGptResponsesModel, parseModelNameWithLevel } from "~/lib/model-level"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import { getTokenCount } from "~/lib/tokenizer"
import { isNullish } from "~/lib/utils"
import {
  createChatCompletions,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
  normalizeChatCompletionsPayloadModel,
} from "~/services/copilot/create-chat-completions"
import {
  createResponses,
  type ResponsesApiResponse,
} from "~/services/copilot/create-responses"

import {
  translateChatCompletionsToResponses,
  translateResponsesStreamToChatStream,
  translateResponsesToChatCompletions,
} from "./responses-translation"

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  let payload = await c.req.json<ChatCompletionsPayload>()
  const { baseModel } = parseModelNameWithLevel(payload.model)
  consola.debug("Request payload:", JSON.stringify(payload).slice(-400))

  // Find the selected model
  const selectedModel = state.models?.data.find(
    (model) => model.id === baseModel,
  )

  // Calculate and display token count
  try {
    if (selectedModel) {
      const tokenCount = await getTokenCount(payload, selectedModel)
      consola.info("Current token count:", tokenCount)
    } else {
      consola.warn("No model selected, skipping token count calculation")
    }
  } catch (error) {
    consola.warn("Failed to calculate token count:", error)
  }

  if (state.manualApprove) await awaitApproval()

  if (isNullish(payload.max_tokens)) {
    payload = {
      ...payload,
      max_tokens: selectedModel?.capabilities.limits.max_output_tokens,
    }
    consola.debug("Set max_tokens to:", JSON.stringify(payload.max_tokens))
  }

  const normalizedPayload = normalizeChatCompletionsPayloadModel(payload)

  if (isGptResponsesModel(baseModel)) {
    const responsesPayload =
      translateChatCompletionsToResponses(normalizedPayload)
    const responses = await createResponses(responsesPayload)

    if (isNonStreamingResponse(responses)) {
      const completionResponse = translateResponsesToChatCompletions(responses)
      consola.debug(
        "GPT translated response:",
        JSON.stringify(completionResponse).slice(-400),
      )
      return c.json(completionResponse)
    }

    return streamSSE(c, async (stream) => {
      for await (const chunk of translateResponsesStreamToChatStream(
        responses,
        normalizedPayload.model,
      )) {
        await stream.writeSSE(chunk)
      }
    })
  }

  const response = await createChatCompletions(normalizedPayload)

  if (isNonStreaming(response)) {
    consola.debug("Non-streaming response:", JSON.stringify(response))
    return c.json(response)
  }

  consola.debug("Streaming response")
  return streamSSE(c, async (stream) => {
    for await (const chunk of response) {
      consola.debug("Streaming chunk:", JSON.stringify(chunk))
      await stream.writeSSE(chunk as SSEMessage)
    }
  })
}

const isNonStreamingResponse = (
  response: Awaited<ReturnType<typeof createResponses>>,
): response is ResponsesApiResponse => !(Symbol.asyncIterator in response)

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")
