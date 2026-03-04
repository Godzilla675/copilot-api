import type { SSEMessage } from "hono/streaming"

import { randomUUID } from "node:crypto"

import type {
  ChatCompletionChunk,
  ChatCompletionResponse,
  ChatCompletionsPayload,
  ContentPart,
  Message,
  ToolCall,
} from "~/services/copilot/create-chat-completions"
import type {
  ResponseInputContentPart,
  ResponseInputMessage,
  ResponsesApiResponse,
  ResponsesFunctionCall,
  ResponsesOutputContentPart,
  ResponsesOutputItem,
  ResponsesPayload,
} from "~/services/copilot/create-responses"

export function translateChatCompletionsToResponses(
  payload: ChatCompletionsPayload,
): ResponsesPayload {
  return {
    model: payload.model,
    input: payload.messages.map((message) => translateMessage(message)),
    stream: payload.stream,
    temperature: payload.temperature,
    top_p: payload.top_p,
    max_output_tokens: payload.max_tokens,
    stop: payload.stop,
    tools: payload.tools as Array<unknown> | null | undefined,
    tool_choice: payload.tool_choice,
    user: payload.user,
    reasoning_effort: payload.reasoning_effort,
    reasoning:
      payload.reasoning
      ?? (payload.reasoning_effort ?
        {
          effort: payload.reasoning_effort,
        }
      : undefined),
  }
}

export function translateResponsesToChatCompletions(
  response: ResponsesApiResponse,
): ChatCompletionResponse {
  const outputItems = response.output ?? []
  const messageContent = extractOutputText(outputItems, response.output_text)
  const toolCalls = extractToolCalls(outputItems)
  const completionTokens = response.usage?.output_tokens ?? 0
  const promptTokens = response.usage?.input_tokens ?? 0

  return {
    id: response.id,
    object: "chat.completion",
    created: response.created_at ?? Math.floor(Date.now() / 1000),
    model: response.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: messageContent.length > 0 ? messageContent : null,
          ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
        },
        logprobs: null,
        finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop",
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens:
        response.usage?.total_tokens ?? promptTokens + completionTokens,
    },
  }
}

export async function* translateResponsesStreamToChatStream(
  responseStream: AsyncIterable<{ data?: string }>,
  model: string,
): AsyncGenerator<SSEMessage> {
  const completionId = randomUUID()
  const created = Math.floor(Date.now() / 1000)
  let hasEmittedContent = false

  for await (const rawEvent of responseStream) {
    if (rawEvent.data === "[DONE]") {
      const endChunk: ChatCompletionChunk = {
        id: completionId,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "stop",
            logprobs: null,
          },
        ],
      }
      yield { data: JSON.stringify(endChunk) }
      yield { data: "[DONE]" }
      return
    }

    if (!rawEvent.data) {
      continue
    }

    const parsedEvent = JSON.parse(rawEvent.data) as {
      type?: string
      delta?: string
    }

    if (
      parsedEvent.type === "response.output_text.delta"
      && typeof parsedEvent.delta === "string"
    ) {
      const chunk: ChatCompletionChunk = {
        id: completionId,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {
              ...(hasEmittedContent ? {} : { role: "assistant" }),
              content: parsedEvent.delta,
            },
            finish_reason: null,
            logprobs: null,
          },
        ],
      }
      hasEmittedContent = true
      yield { data: JSON.stringify(chunk) }
    }
  }
}

function translateMessage(message: Message): ResponseInputMessage {
  let content: ResponseInputMessage["content"]
  if (typeof message.content === "string") {
    content = message.content
  } else if (message.content === null) {
    content = ""
  } else {
    content = message.content.map((part) => translateContentPart(part))
  }

  return {
    role: message.role,
    content,
  }
}

function translateContentPart(part: ContentPart): ResponseInputContentPart {
  if (part.type === "text") {
    return {
      type: "input_text",
      text: part.text,
    }
  }

  return {
    type: "input_image",
    image_url: part.image_url.url,
    detail: part.image_url.detail,
  }
}

function extractOutputText(
  outputItems: Array<ResponsesOutputItem>,
  outputText: string | undefined,
): string {
  if (outputText) {
    return outputText
  }

  const parts = outputItems.flatMap((item) => {
    if (item.type !== "message") {
      return []
    }

    if (typeof item.content === "string") {
      return [item.content]
    }

    if (!Array.isArray(item.content)) {
      return []
    }

    return item.content.flatMap((contentPart) =>
      contentPart.type === "output_text" ? [contentPart.text] : [],
    )
  })

  return parts.join("")
}

function extractToolCalls(
  outputItems: Array<ResponsesOutputItem>,
): Array<ToolCall> {
  const toolCalls: Array<ToolCall> = []
  for (const item of outputItems) {
    if (item.type === "function_call") {
      toolCalls.push(translateFunctionCall(item))
      continue
    }

    if (typeof item.content === "string" || !Array.isArray(item.content)) {
      continue
    }

    for (const contentPart of item.content) {
      if (isResponsesFunctionCall(contentPart)) {
        toolCalls.push(translateFunctionCall(contentPart))
      }
    }
  }

  return toolCalls
}

function translateFunctionCall(functionCall: ResponsesFunctionCall): ToolCall {
  return {
    id: functionCall.call_id ?? functionCall.id ?? randomUUID(),
    type: "function",
    function: {
      name: functionCall.name,
      arguments: functionCall.arguments,
    },
  }
}

function isResponsesFunctionCall(
  value: ResponsesOutputContentPart,
): value is ResponsesFunctionCall {
  return (
    value.type === "function_call"
    && "name" in value
    && typeof value.name === "string"
    && "arguments" in value
    && typeof value.arguments === "string"
  )
}
