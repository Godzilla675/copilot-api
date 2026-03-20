import { describe, expect, test } from "bun:test"

import type { AnthropicMessagesPayload } from "~/routes/messages/anthropic-types"
import type { ResponsesApiResponse } from "~/services/copilot/create-responses"

import {
  createResponsesStreamState,
  translateResponsesStreamEvent,
} from "~/routes/messages/responses-stream-translation"
import {
  THINKING_TEXT,
  translateAnthropicToResponses,
  translateResponsesToAnthropic,
} from "~/routes/messages/responses-translation"

describe("Anthropic messages Responses request translation", () => {
  test("preserves thinking signatures in responses input", () => {
    const payload: AnthropicMessagesPayload = {
      model: "claude-sonnet-4.6",
      messages: [
        {
          role: "assistant",
          content: [
            {
              type: "thinking",
              thinking: "I should reuse the prior reasoning state.",
              signature: "encrypted-thought@rs_123",
            },
            {
              type: "text",
              text: "I already checked that.",
            },
          ],
        },
        {
          role: "user",
          content: "What about the next step?",
        },
      ],
      max_tokens: 512,
      thinking: {
        type: "enabled",
        budget_tokens: 2048,
      },
    }

    const translated = translateAnthropicToResponses(payload)
    expect(translated.include).toEqual(["reasoning.encrypted_content"])
    expect(translated.reasoning).toEqual({ summary: "detailed" })
    expect(Array.isArray(translated.input)).toBe(true)

    if (!Array.isArray(translated.input)) {
      throw new TypeError("Expected translated input to be an array")
    }

    expect(translated.input[0]).toMatchObject({
      type: "reasoning",
      encrypted_content: "encrypted-thought",
      id: "rs_123",
      summary: [
        {
          type: "summary_text",
          text: "I should reuse the prior reasoning state.",
        },
      ],
    })
    expect(translated.input[1]).toMatchObject({
      role: "assistant",
      content: "I already checked that.",
    })
    expect(translated.input[2]).toMatchObject({
      role: "user",
      content: "What about the next step?",
    })
  })

  test("normalizes unsupported Claude subagent models on responses path", () => {
    const translated = translateAnthropicToResponses({
      model: "claude-sonnet-4-20250110",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 64,
      thinking: {
        type: "enabled",
        budget_tokens: 1024,
      },
    })

    expect(translated.model).toBe("claude-sonnet-4")
  })
})

describe("Anthropic messages Responses response translation", () => {
  test("translates responses reasoning output back to anthropic thinking", () => {
    const response: ResponsesApiResponse = {
      id: "resp_123",
      object: "response",
      model: "claude-sonnet-4.6",
      output: [
        {
          type: "reasoning",
          id: "rs_123",
          encrypted_content: "encrypted-thought",
          summary: [
            {
              type: "summary_text",
              text: "I should reuse the prior reasoning state.",
            },
          ],
        },
        {
          type: "message",
          role: "assistant",
          content: [
            {
              type: "output_text",
              text: "Here is the final answer.",
            },
          ],
        },
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 4,
        total_tokens: 14,
      },
    }

    const translated = translateResponsesToAnthropic(response)

    expect(translated.content).toEqual([
      {
        type: "thinking",
        thinking: "I should reuse the prior reasoning state.",
        signature: "encrypted-thought@rs_123",
      },
      {
        type: "text",
        text: "Here is the final answer.",
      },
    ])
    expect(translated.stop_reason).toBe("end_turn")
  })
})

describe("Anthropic messages Responses reasoning stream translation", () => {
  test("streams reasoning summary and signature as anthropic thinking deltas", () => {
    const state = createResponsesStreamState()
    const events = [
      {
        type: "response.created",
        response: {
          id: "resp_123",
          object: "response",
          model: "claude-sonnet-4.6",
          usage: {
            input_tokens: 10,
            output_tokens: 0,
          },
        },
      },
      {
        type: "response.reasoning_summary_text.delta",
        output_index: 0,
        delta: "Tracing the previous reasoning.",
      },
      {
        type: "response.output_item.done",
        output_index: 0,
        item: {
          type: "reasoning",
          id: "rs_123",
          encrypted_content: "encrypted-thought",
        },
      },
      {
        type: "response.completed",
        response: {
          id: "resp_123",
          object: "response",
          model: "claude-sonnet-4.6",
          output: [
            {
              type: "reasoning",
              id: "rs_123",
              encrypted_content: "encrypted-thought",
            },
          ],
          usage: {
            input_tokens: 10,
            output_tokens: 4,
          },
        },
      },
    ].flatMap((event) => translateResponsesStreamEvent(event, state))

    expect(events[0]).toMatchObject({
      type: "message_start",
      message: {
        id: "resp_123",
      },
    })
    expect(events[1]).toEqual({
      type: "content_block_start",
      index: 0,
      content_block: {
        type: "thinking",
        thinking: "",
      },
    })
    expect(events[2]).toEqual({
      type: "content_block_delta",
      index: 0,
      delta: {
        type: "thinking_delta",
        thinking: "Tracing the previous reasoning.",
      },
    })
    expect(events[3]).toEqual({
      type: "content_block_delta",
      index: 0,
      delta: {
        type: "signature_delta",
        signature: "encrypted-thought@rs_123",
      },
    })
    expect(events.at(-2)).toMatchObject({
      type: "message_delta",
      delta: {
        stop_reason: "end_turn",
      },
      usage: {
        input_tokens: 10,
        output_tokens: 4,
      },
    })
    expect(events.at(-1)).toEqual({
      type: "message_stop",
    })
  })

  test("does not inject placeholder thinking text when a signed block reopens", () => {
    const state = createResponsesStreamState()
    const events = [
      {
        type: "response.reasoning_summary_text.delta",
        output_index: 0,
        delta: "Real reasoning text.",
      },
      {
        type: "response.output_text.delta",
        output_index: 1,
        content_index: 0,
        delta: "Visible answer.",
      },
      {
        type: "response.output_item.done",
        output_index: 0,
        item: {
          type: "reasoning",
          id: "rs_789",
          encrypted_content: "opaque-signature",
        },
      },
    ].flatMap((event) => translateResponsesStreamEvent(event, state))

    const placeholderEvents = events.filter(
      (event) =>
        event.type === "content_block_delta"
        && event.delta.type === "thinking_delta"
        && event.delta.thinking === THINKING_TEXT,
    )

    expect(placeholderEvents).toHaveLength(0)
    expect(events.at(-2)).toEqual({
      type: "content_block_start",
      index: 2,
      content_block: {
        type: "thinking",
        thinking: "",
      },
    })
    expect(events.at(-1)).toEqual({
      type: "content_block_delta",
      index: 2,
      delta: {
        type: "signature_delta",
        signature: "opaque-signature@rs_789",
      },
    })
  })
})

describe("Anthropic messages Responses tool stream translation", () => {
  test("reopens tool blocks with a new index and avoids duplicating done args", () => {
    const state = createResponsesStreamState()
    const events = [
      {
        type: "response.output_item.added",
        output_index: 0,
        item: {
          type: "function_call",
          call_id: "call_123",
          name: "get_weather",
        },
      },
      {
        type: "response.function_call_arguments.delta",
        output_index: 0,
        delta: '{"city":"Bos',
      },
      {
        type: "response.output_text.delta",
        output_index: 1,
        content_index: 0,
        delta: "Working on it.",
      },
      {
        type: "response.function_call_arguments.delta",
        output_index: 0,
        delta: 'ton"}',
      },
      {
        type: "response.function_call_arguments.done",
        output_index: 0,
        arguments: '{"city":"Boston"}',
      },
    ].flatMap((event) => translateResponsesStreamEvent(event, state))

    expect(events).toEqual([
      {
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "tool_use",
          id: "call_123",
          name: "get_weather",
          input: {},
        },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: {
          type: "input_json_delta",
          partial_json: '{"city":"Bos',
        },
      },
      {
        type: "content_block_stop",
        index: 0,
      },
      {
        type: "content_block_start",
        index: 1,
        content_block: {
          type: "text",
          text: "",
        },
      },
      {
        type: "content_block_delta",
        index: 1,
        delta: {
          type: "text_delta",
          text: "Working on it.",
        },
      },
      {
        type: "content_block_stop",
        index: 1,
      },
      {
        type: "content_block_start",
        index: 2,
        content_block: {
          type: "tool_use",
          id: "call_123",
          name: "get_weather",
          input: {},
        },
      },
      {
        type: "content_block_delta",
        index: 2,
        delta: {
          type: "input_json_delta",
          partial_json: 'ton"}',
        },
      },
    ])
  })
})

describe("Anthropic messages Responses placeholder translation", () => {
  test("uses placeholder thinking text when reasoning summary is absent", () => {
    const translated = translateResponsesToAnthropic({
      id: "resp_456",
      object: "response",
      model: "claude-sonnet-4.6",
      output: [
        {
          type: "reasoning",
          id: "rs_456",
          encrypted_content: "opaque-thought",
        },
      ],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        total_tokens: 2,
      },
    })

    expect(translated.content[0]).toEqual({
      type: "thinking",
      thinking: THINKING_TEXT,
      signature: "opaque-thought@rs_456",
    })
  })
})
