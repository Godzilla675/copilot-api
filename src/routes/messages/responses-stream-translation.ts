import type { ResponsesApiResponse } from "~/services/copilot/create-responses"

import type { AnthropicStreamEventData } from "./anthropic-types"

import { THINKING_TEXT } from "./responses-translation"

export interface ResponsesStreamState {
  messageStartSent: boolean
  nextContentBlockIndex: number
  openBlockKey?: string
  openBlockIndex?: number
  blockIndexByKey: Map<string, number>
  blockHasDelta: Set<number>
}

export function createResponsesStreamState(): ResponsesStreamState {
  return {
    messageStartSent: false,
    nextContentBlockIndex: 0,
    blockIndexByKey: new Map(),
    blockHasDelta: new Set(),
  }
}

export function translateResponsesStreamEvent(
  parsedEvent: { type?: string; [key: string]: unknown },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  switch (parsedEvent.type) {
    case "response.created": {
      return handleCreated(parsedEvent, state)
    }
    case "response.reasoning_summary_text.delta": {
      return handleThinkingDelta(parsedEvent, state)
    }
    case "response.output_item.done": {
      return handleOutputItemDone(parsedEvent, state)
    }
    case "response.output_text.delta": {
      return handleTextDelta(parsedEvent, state)
    }
    case "response.output_item.added": {
      return handleToolAdded(parsedEvent, state)
    }
    case "response.function_call_arguments.delta": {
      return handleToolArgumentsDelta(parsedEvent, state)
    }
    case "response.function_call_arguments.done": {
      return handleToolArgumentsDone(parsedEvent, state)
    }
    case "response.completed": {
      return handleCompleted(parsedEvent, state)
    }
    default: {
      return []
    }
  }
}

function handleCreated(
  parsedEvent: { response?: unknown },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  if (state.messageStartSent || !isResponseObject(parsedEvent.response)) {
    return []
  }

  state.messageStartSent = true
  const cachedTokens =
    parsedEvent.response.usage?.input_tokens_details?.cached_tokens

  return [
    {
      type: "message_start",
      message: {
        id: parsedEvent.response.id,
        type: "message",
        role: "assistant",
        content: [],
        model: parsedEvent.response.model,
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens:
            (parsedEvent.response.usage?.input_tokens ?? 0)
            - (cachedTokens ?? 0),
          output_tokens: 0,
          ...(cachedTokens !== undefined && {
            cache_read_input_tokens: cachedTokens,
          }),
        },
      },
    },
  ]
}

function handleThinkingDelta(
  parsedEvent: { output_index?: unknown; delta?: unknown },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  if (
    typeof parsedEvent.output_index !== "number"
    || typeof parsedEvent.delta !== "string"
  ) {
    return []
  }

  const events: Array<AnthropicStreamEventData> = []
  const blockIndex = openBlock({
    state,
    key: `thinking:${parsedEvent.output_index}`,
    contentBlock: {
      type: "thinking",
      thinking: "",
    },
    events,
  })

  events.push({
    type: "content_block_delta",
    index: blockIndex,
    delta: {
      type: "thinking_delta",
      thinking: parsedEvent.delta,
    },
  })
  state.blockHasDelta.add(blockIndex)
  return events
}

function handleOutputItemDone(
  parsedEvent: { output_index?: unknown; item?: unknown },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  if (
    typeof parsedEvent.output_index !== "number"
    || !isRecord(parsedEvent.item)
  ) {
    return []
  }

  if (
    parsedEvent.item.type !== "reasoning"
    || typeof parsedEvent.item.encrypted_content !== "string"
  ) {
    return []
  }

  const events: Array<AnthropicStreamEventData> = []
  const blockIndex = openBlock({
    state,
    key: `thinking:${parsedEvent.output_index}`,
    contentBlock: {
      type: "thinking",
      thinking: "",
    },
    events,
  })

  if (!state.blockHasDelta.has(blockIndex)) {
    events.push({
      type: "content_block_delta",
      index: blockIndex,
      delta: {
        type: "thinking_delta",
        thinking: THINKING_TEXT,
      },
    })
  }

  events.push({
    type: "content_block_delta",
    index: blockIndex,
    delta: {
      type: "signature_delta",
      signature:
        typeof parsedEvent.item.id === "string" ?
          `${parsedEvent.item.encrypted_content}@${parsedEvent.item.id}`
        : parsedEvent.item.encrypted_content,
    },
  })
  state.blockHasDelta.add(blockIndex)
  return events
}

function handleTextDelta(
  parsedEvent: {
    output_index?: unknown
    content_index?: unknown
    delta?: unknown
  },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  if (
    typeof parsedEvent.output_index !== "number"
    || typeof parsedEvent.content_index !== "number"
    || typeof parsedEvent.delta !== "string"
  ) {
    return []
  }

  const events: Array<AnthropicStreamEventData> = []
  const blockIndex = openBlock({
    state,
    key: `text:${parsedEvent.output_index}:${parsedEvent.content_index}`,
    contentBlock: { type: "text", text: "" },
    events,
  })

  events.push({
    type: "content_block_delta",
    index: blockIndex,
    delta: {
      type: "text_delta",
      text: parsedEvent.delta,
    },
  })
  state.blockHasDelta.add(blockIndex)
  return events
}

function handleToolAdded(
  parsedEvent: {
    output_index?: unknown
    item?: unknown
  },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  if (
    typeof parsedEvent.output_index !== "number"
    || !isRecord(parsedEvent.item)
    || parsedEvent.item.type !== "function_call"
    || typeof parsedEvent.item.call_id !== "string"
    || typeof parsedEvent.item.name !== "string"
  ) {
    return []
  }

  const events: Array<AnthropicStreamEventData> = []
  openBlock({
    state,
    key: `tool:${parsedEvent.output_index}`,
    contentBlock: {
      type: "tool_use",
      id: parsedEvent.item.call_id,
      name: parsedEvent.item.name,
      input: {},
    },
    events,
  })
  return events
}

function handleToolArgumentsDelta(
  parsedEvent: {
    output_index?: unknown
    delta?: unknown
  },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  if (
    typeof parsedEvent.output_index !== "number"
    || typeof parsedEvent.delta !== "string"
  ) {
    return []
  }

  const blockIndex = state.blockIndexByKey.get(
    `tool:${parsedEvent.output_index}`,
  )
  if (blockIndex === undefined) {
    return []
  }

  return [
    {
      type: "content_block_delta",
      index: blockIndex,
      delta: {
        type: "input_json_delta",
        partial_json: parsedEvent.delta,
      },
    },
  ]
}

function handleToolArgumentsDone(
  parsedEvent: {
    output_index?: unknown
    arguments?: unknown
  },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  if (
    typeof parsedEvent.output_index !== "number"
    || typeof parsedEvent.arguments !== "string"
  ) {
    return []
  }

  const blockIndex = state.blockIndexByKey.get(
    `tool:${parsedEvent.output_index}`,
  )
  if (blockIndex === undefined || state.blockHasDelta.has(blockIndex)) {
    return []
  }

  state.blockHasDelta.add(blockIndex)
  return [
    {
      type: "content_block_delta",
      index: blockIndex,
      delta: {
        type: "input_json_delta",
        partial_json: parsedEvent.arguments,
      },
    },
  ]
}

function handleCompleted(
  parsedEvent: { response?: unknown },
  state: ResponsesStreamState,
): Array<AnthropicStreamEventData> {
  if (!isResponseObject(parsedEvent.response)) {
    return []
  }

  const events: Array<AnthropicStreamEventData> = []
  closeOpenBlock(state, events)
  const cachedTokens =
    parsedEvent.response.usage?.input_tokens_details?.cached_tokens

  events.push(
    {
      type: "message_delta",
      delta: {
        stop_reason:
          (
            parsedEvent.response.output?.some(
              (item) => item.type === "function_call",
            )
          ) ?
            "tool_use"
          : "end_turn",
        stop_sequence: null,
      },
      usage: {
        input_tokens:
          (parsedEvent.response.usage?.input_tokens ?? 0) - (cachedTokens ?? 0),
        output_tokens: parsedEvent.response.usage?.output_tokens ?? 0,
        ...(cachedTokens !== undefined && {
          cache_read_input_tokens: cachedTokens,
        }),
      },
    },
    {
      type: "message_stop",
    },
  )

  return events
}

function openBlock(params: {
  state: ResponsesStreamState
  key: string
  contentBlock:
    | { type: "thinking"; thinking: string }
    | { type: "text"; text: string }
    | {
        type: "tool_use"
        id: string
        name: string
        input: Record<string, unknown>
      }
  events: Array<AnthropicStreamEventData>
}): number {
  const { state, key, contentBlock, events } = params
  let blockIndex = state.blockIndexByKey.get(key)
  if (blockIndex === undefined) {
    blockIndex = state.nextContentBlockIndex
    state.nextContentBlockIndex += 1
    state.blockIndexByKey.set(key, blockIndex)
  }

  if (state.openBlockKey === key) {
    return blockIndex
  }

  closeOpenBlock(state, events)
  events.push({
    type: "content_block_start",
    index: blockIndex,
    content_block: contentBlock,
  })
  state.openBlockKey = key
  state.openBlockIndex = blockIndex
  return blockIndex
}

function closeOpenBlock(
  state: ResponsesStreamState,
  events: Array<AnthropicStreamEventData>,
): void {
  if (state.openBlockIndex === undefined) {
    return
  }

  events.push({
    type: "content_block_stop",
    index: state.openBlockIndex,
  })
  state.openBlockKey = undefined
  state.openBlockIndex = undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function isResponseObject(value: unknown): value is ResponsesApiResponse {
  return (
    isRecord(value)
    && typeof value.id === "string"
    && typeof value.model === "string"
  )
}
