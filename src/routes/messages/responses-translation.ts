import type {
  ResponseInputItem,
  ResponseInputContentPart,
  ResponsesApiResponse,
  ResponsesOutputItem,
  ResponsesPayload,
  ResponsesReasoningItem,
} from "~/services/copilot/create-responses"

import type {
  AnthropicAssistantContentBlock,
  AnthropicAssistantMessage,
  AnthropicMessage,
  AnthropicMessagesPayload,
  AnthropicResponse,
  AnthropicTextBlock,
  AnthropicThinkingBlock,
  AnthropicTool,
  AnthropicToolResultBlock,
  AnthropicToolUseBlock,
  AnthropicUserContentBlock,
  AnthropicUserMessage,
} from "./anthropic-types"

import { translateModelName } from "./non-stream-translation"

export const THINKING_TEXT = "Thinking..."

type InputItem = ResponseInputItem

export function translateAnthropicToResponses(
  payload: AnthropicMessagesPayload,
): ResponsesPayload {
  const input = [
    ...translateSystemPrompt(payload.system),
    ...payload.messages.flatMap((message) => translateMessage(message)),
  ]

  return {
    model: translateModelName(payload.model),
    input,
    stream: payload.stream,
    temperature: payload.temperature,
    top_p: payload.top_p,
    max_output_tokens: payload.max_tokens,
    stop: payload.stop_sequences,
    tools: translateAnthropicTools(payload.tools) as Array<unknown> | undefined,
    tool_choice: translateAnthropicToolChoice(payload.tool_choice),
    user: payload.metadata?.user_id,
    reasoning: {
      summary: "detailed",
    },
    include: ["reasoning.encrypted_content"],
  }
}

export function translateResponsesToAnthropic(
  response: ResponsesApiResponse,
): AnthropicResponse {
  const content = response.output?.flatMap((item) => mapOutputItem(item)) ?? []

  const anthropicContent =
    content.length > 0 || !response.output_text ?
      content
    : [{ type: "text", text: response.output_text }]

  return {
    id: response.id,
    type: "message",
    role: "assistant",
    content: anthropicContent,
    model: response.model,
    stop_reason: getStopReason(response),
    stop_sequence: null,
    usage: {
      input_tokens:
        (response.usage?.input_tokens ?? 0)
        - (response.usage?.input_tokens_details?.cached_tokens ?? 0),
      output_tokens: response.usage?.output_tokens ?? 0,
      ...(response.usage?.input_tokens_details?.cached_tokens !== undefined && {
        cache_read_input_tokens:
          response.usage.input_tokens_details.cached_tokens,
      }),
    },
  }
}

function translateSystemPrompt(
  system: AnthropicMessagesPayload["system"],
): Array<InputItem> {
  if (!system) {
    return []
  }

  if (typeof system === "string") {
    return [{ role: "system", content: system }]
  }

  return [
    {
      role: "system",
      content: system.map((block) => block.text).join("\n\n"),
    },
  ]
}

function translateMessage(message: AnthropicMessage): Array<InputItem> {
  return message.role === "user" ?
      translateUserMessage(message)
    : translateAssistantMessage(message)
}

function translateUserMessage(message: AnthropicUserMessage): Array<InputItem> {
  if (typeof message.content === "string") {
    return [{ role: "user", content: message.content }]
  }

  const toolResults = message.content.filter(
    (block): block is AnthropicToolResultBlock => block.type === "tool_result",
  )
  const otherBlocks = message.content.filter(
    (block) => block.type !== "tool_result",
  )

  const translated = toolResults.map((block) => ({
    role: "tool" as const,
    tool_call_id: block.tool_use_id,
    content: typeof block.content === "string" ? block.content : "",
  }))

  if (otherBlocks.length === 0) {
    return translated
  }

  return [
    ...translated,
    {
      role: "user",
      content: mapUserContent(otherBlocks),
    },
  ]
}

function translateAssistantMessage(
  message: AnthropicAssistantMessage,
): Array<InputItem> {
  if (typeof message.content === "string") {
    return [{ role: "assistant", content: message.content }]
  }

  const reasoningItems = message.content.flatMap((block) =>
    block.type === "thinking" && block.signature ?
      [translateThinkingBlock(block)]
    : [],
  )
  const text = message.content
    .filter(
      (block): block is AnthropicTextBlock | AnthropicThinkingBlock =>
        block.type === "text"
        || (block.type === "thinking" && !block.signature),
    )
    .map((block) => (block.type === "text" ? block.text : block.thinking))
    .join("\n\n")
  const toolCalls = message.content
    .filter(
      (block): block is AnthropicToolUseBlock => block.type === "tool_use",
    )
    .map((block) => ({
      id: block.id,
      type: "function" as const,
      function: {
        name: block.name,
        arguments: JSON.stringify(block.input),
      },
    }))

  if (text.length === 0 && toolCalls.length === 0) {
    return reasoningItems
  }

  return [
    ...reasoningItems,
    {
      role: "assistant",
      content: text,
      ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
    },
  ]
}

function mapUserContent(
  content: Array<AnthropicUserContentBlock>,
): string | Array<ResponseInputContentPart> {
  const hasImage = content.some((block) => block.type === "image")
  if (!hasImage) {
    return content
      .filter((block): block is AnthropicTextBlock => block.type === "text")
      .map((block) => block.text)
      .join("\n\n")
  }

  return content.flatMap((block) => {
    if (block.type === "text") {
      return [{ type: "input_text" as const, text: block.text }]
    }

    if (block.type === "image") {
      return [
        {
          type: "input_image" as const,
          image_url: `data:${block.source.media_type};base64,${block.source.data}`,
          detail: "auto" as const,
        },
      ]
    }

    return []
  })
}

function translateThinkingBlock(
  block: AnthropicThinkingBlock & { signature: string },
) {
  const { encryptedContent, id } = parseThinkingSignature(block.signature)
  return {
    type: "reasoning" as const,
    encrypted_content: encryptedContent,
    ...(id ? { id } : {}),
    summary:
      block.thinking && block.thinking !== THINKING_TEXT ?
        [{ type: "summary_text" as const, text: block.thinking }]
      : [],
  }
}

function mapOutputItem(
  item: ResponsesOutputItem,
): Array<AnthropicAssistantContentBlock> {
  if (item.type === "reasoning") {
    return mapReasoningItem(item)
  }

  if (item.type === "function_call") {
    if (!item.call_id || !item.name) {
      return []
    }

    return [
      {
        type: "tool_use",
        id: item.call_id,
        name: item.name,
        input: parseFunctionArguments(item.arguments),
      },
    ]
  }

  const text = extractMessageText(item.content)
  return text.length > 0 ? [{ type: "text", text }] : []
}

function mapReasoningItem(
  item: ResponsesReasoningItem,
): Array<AnthropicAssistantContentBlock> {
  const signature = buildThinkingSignature(item.encrypted_content, item.id)
  if (!signature) {
    return []
  }

  return [
    {
      type: "thinking",
      thinking: extractThinkingText(item.summary),
      signature,
    },
  ]
}

function parseThinkingSignature(signature: string): {
  encryptedContent: string
  id: string | undefined
} {
  const separatorIndex = signature.lastIndexOf("@")
  if (separatorIndex <= 0 || separatorIndex === signature.length - 1) {
    return { encryptedContent: signature, id: undefined }
  }

  return {
    encryptedContent: signature.slice(0, separatorIndex),
    id: signature.slice(separatorIndex + 1),
  }
}

function extractThinkingText(
  summary: Array<{ type: string; text?: string }> | undefined,
): string {
  const text = summary
    ?.filter((block) => typeof block.text === "string")
    .map((block) => block.text)
    .join("")
    .trim()

  return text && text.length > 0 ? text : THINKING_TEXT
}

function buildThinkingSignature(
  encryptedContent: string | undefined,
  id: string | undefined,
): string | undefined {
  if (!encryptedContent) {
    return undefined
  }

  return id ? `${encryptedContent}@${id}` : encryptedContent
}

function extractMessageText(
  content: string | Array<{ type: string; text?: string }>,
): string {
  if (typeof content === "string") {
    return content
  }

  return content
    .filter((part) => part.type === "output_text" || part.type === "text")
    .map((part) => part.text ?? "")
    .join("")
}

function parseFunctionArguments(
  argumentsText: string | undefined,
): Record<string, unknown> {
  if (!argumentsText) {
    return {}
  }

  try {
    return JSON.parse(argumentsText) as Record<string, unknown>
  } catch {
    return {}
  }
}

function translateAnthropicTools(
  anthropicTools: Array<AnthropicTool> | undefined,
):
  | Array<{
      type: "function"
      function: {
        name: string
        description?: string
        parameters: Record<string, unknown>
      }
    }>
  | undefined {
  if (!anthropicTools) {
    return undefined
  }

  return anthropicTools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
    },
  }))
}

function translateAnthropicToolChoice(
  anthropicToolChoice: AnthropicMessagesPayload["tool_choice"],
):
  | "auto"
  | "required"
  | "none"
  | { type: "function"; function: { name: string } }
  | undefined {
  if (!anthropicToolChoice) {
    return undefined
  }

  switch (anthropicToolChoice.type) {
    case "auto": {
      return "auto"
    }
    case "any": {
      return "required"
    }
    case "tool": {
      return anthropicToolChoice.name ?
          { type: "function", function: { name: anthropicToolChoice.name } }
        : "auto"
    }
    case "none": {
      return "none"
    }
    default: {
      return undefined
    }
  }
}

function getStopReason(
  response: ResponsesApiResponse,
): AnthropicResponse["stop_reason"] {
  return response.output?.some((item) => item.type === "function_call") ?
      "tool_use"
    : "end_turn"
}
