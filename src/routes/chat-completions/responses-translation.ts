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

// eslint-disable-next-line max-lines-per-function, complexity
export async function* translateResponsesStreamToChatStream(
  responseStream: AsyncIterable<{ data?: string }>,
  model: string,
): AsyncGenerator<SSEMessage> {
  const completionId = randomUUID()
  const created = Math.floor(Date.now() / 1000)
  let hasEmittedRole = false
  let finishReason: ChatChunkFinishReason = "stop"
  let hasEmittedTerminalChunk = false
  let nextToolCallIndex = 0
  const toolCallStates = new Map<string, StreamToolCallState>()

  for await (const rawEvent of responseStream) {
    if (rawEvent.data === "[DONE]") {
      if (!hasEmittedTerminalChunk) {
        yield {
          data: JSON.stringify(
            createChatChunk(completionId, created, model, {}, finishReason),
          ),
        }
      }
      yield { data: "[DONE]" }
      return
    }

    if (!rawEvent.data) {
      continue
    }

    const parsedEvent = JSON.parse(rawEvent.data) as {
      type?: string
      [key: string]: unknown
    }

    if (
      parsedEvent.type === "response.output_text.delta"
      && typeof parsedEvent.delta === "string"
    ) {
      const delta = withAssistantRole(hasEmittedRole, {
        content: parsedEvent.delta,
      })
      hasEmittedRole = true
      yield {
        data: JSON.stringify(
          createChatChunk(completionId, created, model, delta, null),
        ),
      }
      continue
    }

    if (
      parsedEvent.type === "response.output_item.added"
      && typeof parsedEvent.output_index === "number"
      && isStreamOutputItem(parsedEvent.item)
      && isResponsesFunctionCallItem(parsedEvent.item)
    ) {
      finishReason = "tool_calls"
      const state = getOrCreateToolCallState(
        toolCallStates,
        parsedEvent.item.id
          ?? parsedEvent.item.call_id
          ?? String(parsedEvent.output_index),
        parsedEvent.output_index,
        parsedEvent.item,
        () => nextToolCallIndex++,
      )
      const delta = withAssistantRole(hasEmittedRole, {
        tool_calls: [
          {
            index: state.chatIndex,
            id: state.id,
            type: "function",
            function: {
              name: state.name,
              arguments: "",
            },
          },
        ],
      })
      hasEmittedRole = true
      state.hasEmittedInitialChunk = true
      yield {
        data: JSON.stringify(
          createChatChunk(completionId, created, model, delta, null),
        ),
      }
      continue
    }

    if (
      parsedEvent.type === "response.function_call_arguments.delta"
      && typeof parsedEvent.item_id === "string"
      && typeof parsedEvent.output_index === "number"
      && typeof parsedEvent.call_id === "string"
      && typeof parsedEvent.delta === "string"
    ) {
      finishReason = "tool_calls"
      const state = getOrCreateToolCallState(
        toolCallStates,
        parsedEvent.item_id,
        parsedEvent.output_index,
        {
          type: "function_call",
          call_id: parsedEvent.call_id,
        },
        () => nextToolCallIndex++,
      )
      state.arguments += parsedEvent.delta
      const delta = withAssistantRole(hasEmittedRole, {
        tool_calls: [
          {
            index: state.chatIndex,
            function: {
              arguments: parsedEvent.delta,
            },
          },
        ],
      })
      hasEmittedRole = true
      yield {
        data: JSON.stringify(
          createChatChunk(completionId, created, model, delta, null),
        ),
      }
      continue
    }

    if (
      parsedEvent.type === "response.function_call_arguments.done"
      && typeof parsedEvent.item_id === "string"
      && typeof parsedEvent.output_index === "number"
      && typeof parsedEvent.call_id === "string"
      && typeof parsedEvent.arguments === "string"
    ) {
      finishReason = "tool_calls"
      const state = getOrCreateToolCallState(
        toolCallStates,
        parsedEvent.item_id,
        parsedEvent.output_index,
        {
          type: "function_call",
          call_id: parsedEvent.call_id,
          arguments: parsedEvent.arguments,
        },
        () => nextToolCallIndex++,
      )
      const remainder =
        parsedEvent.arguments.startsWith(state.arguments) ?
          parsedEvent.arguments.slice(state.arguments.length)
        : parsedEvent.arguments
      state.arguments = parsedEvent.arguments
      if (remainder.length === 0) {
        continue
      }

      const delta = withAssistantRole(hasEmittedRole, {
        tool_calls: [
          {
            index: state.chatIndex,
            function: {
              arguments: remainder,
            },
          },
        ],
      })
      hasEmittedRole = true
      yield {
        data: JSON.stringify(
          createChatChunk(completionId, created, model, delta, null),
        ),
      }
      continue
    }

    if (
      parsedEvent.type === "response.output_item.done"
      && typeof parsedEvent.output_index === "number"
      && isStreamOutputItem(parsedEvent.item)
      && isResponsesFunctionCallItem(parsedEvent.item)
    ) {
      finishReason = "tool_calls"
      const itemId =
        parsedEvent.item.id
        ?? parsedEvent.item.call_id
        ?? String(parsedEvent.output_index)
      const state = getOrCreateToolCallState(
        toolCallStates,
        itemId,
        parsedEvent.output_index,
        parsedEvent.item,
        () => nextToolCallIndex++,
      )
      if (state.name === undefined && parsedEvent.item.name) {
        state.name = parsedEvent.item.name
      }

      if (!state.hasEmittedInitialChunk) {
        const delta = withAssistantRole(hasEmittedRole, {
          tool_calls: [
            {
              index: state.chatIndex,
              id: state.id,
              type: "function",
              function: {
                name: state.name,
                arguments: "",
              },
            },
          ],
        })
        hasEmittedRole = true
        state.hasEmittedInitialChunk = true
        yield {
          data: JSON.stringify(
            createChatChunk(completionId, created, model, delta, null),
          ),
        }
      }

      const finalArguments = parsedEvent.item.arguments ?? ""
      const remainder =
        finalArguments.startsWith(state.arguments) ?
          finalArguments.slice(state.arguments.length)
        : finalArguments
      state.arguments = finalArguments
      if (remainder.length === 0) {
        continue
      }

      const delta = withAssistantRole(hasEmittedRole, {
        tool_calls: [
          {
            index: state.chatIndex,
            function: {
              arguments: remainder,
            },
          },
        ],
      })
      hasEmittedRole = true
      yield {
        data: JSON.stringify(
          createChatChunk(completionId, created, model, delta, null),
        ),
      }
      continue
    }

    if (parsedEvent.type === "response.completed") {
      hasEmittedTerminalChunk = true
      yield {
        data: JSON.stringify(
          createChatChunk(completionId, created, model, {}, finishReason),
        ),
      }
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
    ...(message.name ? { name: message.name } : {}),
    ...(message.tool_call_id ? { tool_call_id: message.tool_call_id } : {}),
    ...(message.tool_calls ?
      {
        tool_calls: message.tool_calls.map((toolCall) =>
          translateToolCall(toolCall),
        ),
      }
    : {}),
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

function translateToolCall(
  toolCall: ToolCall,
): NonNullable<ResponseInputMessage["tool_calls"]>[number] {
  return {
    ...toolCall,
    function: {
      ...toolCall.function,
    },
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
      contentPart.type === "output_text" || contentPart.type === "text" ?
        [contentPart.text]
      : [],
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

    if (
      item.type !== "message"
      || typeof item.content === "string"
      || !Array.isArray(item.content)
    ) {
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
      arguments: functionCall.arguments ?? "",
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
  )
}

function isResponsesFunctionCallItem(
  value: ResponsesOutputItem | StreamOutputItem,
): value is ResponsesFunctionCall {
  return value.type === "function_call" && typeof value.name === "string"
}

function isStreamOutputItem(value: unknown): value is StreamOutputItem {
  return (
    typeof value === "object"
    && value !== null
    && "type" in value
    && typeof value.type === "string"
  )
}

// eslint-disable-next-line max-params
function createChatChunk(
  id: string,
  created: number,
  model: string,
  delta: ChatChunkDelta,
  finishReason: ChatChunkFinishReason,
): ChatCompletionChunk {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason,
        logprobs: null,
      },
    ],
  }
}

function withAssistantRole(
  hasEmittedRole: boolean,
  delta: ChatChunkDelta,
): ChatChunkDelta {
  return hasEmittedRole ? delta : { role: "assistant", ...delta }
}

// eslint-disable-next-line max-params
function getOrCreateToolCallState(
  toolCallStates: Map<string, StreamToolCallState>,
  key: string,
  outputIndex: number,
  item: Partial<ResponsesFunctionCall>,
  getNextToolCallIndex: () => number,
): StreamToolCallState {
  const existingState = toolCallStates.get(key)
  if (existingState) {
    if (existingState.name === undefined && item.name) {
      existingState.name = item.name
    }
    return existingState
  }

  const nextState: StreamToolCallState = {
    arguments: item.arguments ?? "",
    chatIndex: getNextToolCallIndex(),
    hasEmittedInitialChunk: false,
    id: item.call_id ?? item.id ?? randomUUID(),
    name: item.name,
    outputIndex,
  }
  toolCallStates.set(key, nextState)
  return nextState
}

type ChatChunkDelta = ChatCompletionChunk["choices"][number]["delta"]
type ChatChunkFinishReason =
  ChatCompletionChunk["choices"][number]["finish_reason"]

interface StreamToolCallState {
  arguments: string
  chatIndex: number
  hasEmittedInitialChunk: boolean
  id: string
  name?: string
  outputIndex: number
}

type StreamOutputItem =
  | ResponsesFunctionCall
  | {
      type: "message"
      id?: string
      role?: "assistant" | "user" | "system" | "tool"
      content?: Array<unknown>
      object?: string
      status?: string
    }
