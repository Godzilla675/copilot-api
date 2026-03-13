import { describe, expect, test } from "bun:test"

import type { ChatCompletionChunk } from "~/services/copilot/create-chat-completions"
import type { ResponseInputMessage } from "~/services/copilot/create-responses"
import type { Model } from "~/services/copilot/get-models"

import {
  getModelLevelsForModel,
  parseModelNameWithLevel,
} from "~/lib/model-level"
import {
  translateChatCompletionsToResponses,
  translateResponsesStreamToChatStream,
  translateResponsesToChatCompletions,
} from "~/routes/chat-completions/responses-translation"
import { expandModelList } from "~/routes/models/route"
import { normalizeChatCompletionsPayloadModel } from "~/services/copilot/create-chat-completions"

describe("model(level) parsing and mapping", () => {
  test("parses suffixed model name", () => {
    expect(parseModelNameWithLevel("gpt-5.3-codex(high)")).toEqual({
      baseModel: "gpt-5.3-codex",
      level: "high",
    })
  })

  test("keeps plain model untouched", () => {
    expect(parseModelNameWithLevel("claude-sonnet-4.6")).toEqual({
      baseModel: "claude-sonnet-4.6",
      level: undefined,
    })
  })

  test("maps GPT-5.4 suffix level to reasoning_effort", () => {
    const payload = normalizeChatCompletionsPayloadModel({
      model: "gpt-5.4(xhigh)",
      messages: [{ role: "user", content: "hi" }],
    })

    expect(payload.model).toBe("gpt-5.4")
    expect(payload.reasoning_effort).toBe("xhigh")
  })

  test("maps claude suffix level while preserving thinking fields", () => {
    const payload = normalizeChatCompletionsPayloadModel({
      model: "claude-opus-4.6(high)",
      messages: [{ role: "user", content: "hi" }],
      thinking: { budget_tokens: 4096 },
    })

    expect(payload.model).toBe("claude-opus-4.6")
    expect(payload.reasoning_effort).toBe("high")
    expect(payload.thinking).toEqual({
      budget_tokens: 4096,
      effort: "high",
      type: "enabled",
    })
  })
})

describe("chat/responses translation", () => {
  test("translates chat payload to responses payload", () => {
    const translated = translateChatCompletionsToResponses({
      model: "gpt-5.4",
      messages: [{ role: "user", content: "Hello" }],
      max_tokens: 128,
      reasoning_effort: "medium",
    })

    expect(translated).toMatchObject({
      model: "gpt-5.4",
      input: [{ role: "user", content: "Hello" }],
      max_output_tokens: 128,
      reasoning_effort: "medium",
      reasoning: { effort: "medium" },
    })
  })

  test("translates responses payload back to chat completion", () => {
    const translated = translateResponsesToChatCompletions({
      id: "resp_123",
      object: "response",
      created_at: 123,
      model: "gpt-5.3-codex",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hello from responses" }],
        },
      ],
      usage: { input_tokens: 2, output_tokens: 4, total_tokens: 6 },
    })

    expect(translated.object).toBe("chat.completion")
    expect(translated.choices[0]?.message.content).toBe("Hello from responses")
    expect(translated.usage).toEqual({
      prompt_tokens: 2,
      completion_tokens: 4,
      total_tokens: 6,
    })
  })

  test("preserves tool metadata when translating chat payload", () => {
    const translated = translateChatCompletionsToResponses({
      model: "gpt-5.4",
      messages: [
        {
          role: "assistant",
          content: null,
          name: "planner",
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "get_weather",
                arguments: '{"city":"Boston"}',
              },
            },
          ],
        },
        {
          role: "tool",
          content: '{"temperature":72}',
          tool_call_id: "call_123",
        },
      ],
    })

    expect(Array.isArray(translated.input)).toBe(true)
    if (!Array.isArray(translated.input)) {
      throw new TypeError(
        "Expected translated input to be an array of messages",
      )
    }
    const input: Array<ResponseInputMessage> = translated.input
    expect(input[0]).toMatchObject({
      role: "assistant",
      content: "",
      name: "planner",
      tool_calls: [
        {
          id: "call_123",
          type: "function",
          function: {
            name: "get_weather",
            arguments: '{"city":"Boston"}',
          },
        },
      ],
    })
    expect(input[1]).toMatchObject({
      role: "tool",
      content: '{"temperature":72}',
      tool_call_id: "call_123",
    })
  })

  test("translates text streaming events back to chat chunks", async () => {
    const translated = await collectStreamChunks(
      translateResponsesStreamToChatStream(
        asResponseStream([
          {
            type: "response.output_item.added",
            output_index: 0,
            item: {
              id: "msg_123",
              type: "message",
              status: "in_progress",
              role: "assistant",
              content: [],
            },
          },
          {
            type: "response.output_text.delta",
            item_id: "msg_123",
            output_index: 0,
            content_index: 0,
            delta: "Hello",
          },
          {
            type: "response.completed",
            response: {
              id: "resp_123",
              object: "response",
              model: "gpt-5.4",
            },
          },
        ]),
        "gpt-5.4",
      ),
    )

    expect(translated[0]?.choices[0]?.delta).toEqual({
      role: "assistant",
      content: "Hello",
    })
    expect(translated[1]?.choices[0]?.finish_reason).toBe("stop")
  })

  test("translates streamed function call events back to chat chunks", async () => {
    const translated = await collectStreamChunks(
      translateResponsesStreamToChatStream(
        asResponseStream([
          {
            type: "response.output_item.added",
            output_index: 0,
            item: {
              id: "fc_123",
              type: "function_call",
              call_id: "call_123",
              name: "get_weather",
            },
          },
          {
            type: "response.function_call_arguments.delta",
            item_id: "fc_123",
            output_index: 0,
            call_id: "call_123",
            delta: '{"location":"San"',
          },
          {
            type: "response.completed",
            response: {
              id: "resp_123",
              object: "response",
              model: "gpt-5.4",
            },
          },
        ]),
        "gpt-5.4",
      ),
    )

    expect(translated[0]?.choices[0]?.delta).toEqual({
      role: "assistant",
      tool_calls: [
        {
          index: 0,
          id: "call_123",
          type: "function",
          function: {
            name: "get_weather",
            arguments: "",
          },
        },
      ],
    })
    expect(translated[1]?.choices[0]?.delta).toEqual({
      tool_calls: [
        {
          index: 0,
          function: {
            arguments: '{"location":"San"',
          },
        },
      ],
    })
    expect(translated[2]?.choices[0]?.finish_reason).toBe("tool_calls")
  })
})

describe("model listing expansion", () => {
  test("includes required level-suffixed variants", () => {
    const models = expandModelList([
      makeModel("gpt-5.4"),
      makeModel("gpt-5.3-codex"),
      makeModel("claude-opus-4.6"),
      makeModel("claude-opus-4.6-fast"),
      makeModel("claude-sonnet-4.6"),
      makeModel("gpt-4.1"),
    ])
    const ids = models.map((model) => model.id)

    expect(ids).toContain("gpt-5.4")
    expect(ids).toContain("gpt-5.3-codex")
    for (const level of getModelLevelsForModel("gpt-5.4") ?? []) {
      expect(ids).toContain(`gpt-5.4(${level})`)
    }
    for (const level of getModelLevelsForModel("gpt-5.3-codex") ?? []) {
      expect(ids).toContain(`gpt-5.3-codex(${level})`)
    }
    for (const level of getModelLevelsForModel("claude-opus-4.6") ?? []) {
      expect(ids).toContain(`claude-opus-4.6(${level})`)
      expect(ids).toContain(`claude-opus-4.6-fast(${level})`)
      expect(ids).toContain(`claude-sonnet-4.6(${level})`)
    }
    expect(ids).toContain("gpt-4.1")
  })
})

function makeModel(id: string): Model {
  return {
    id,
    name: id,
    object: "model",
    model_picker_enabled: true,
    preview: false,
    vendor: "test",
    version: "1",
    capabilities: {
      family: "test",
      limits: {},
      object: "model_capabilities",
      supports: {},
      tokenizer: "test",
      type: "chat",
    },
  }
}

async function collectStreamChunks(
  stream: AsyncIterable<{ data?: string | Promise<string> }>,
): Promise<Array<ChatCompletionChunk>> {
  const chunks: Array<ChatCompletionChunk> = []
  for await (const event of stream) {
    const data = await event.data
    if (data === "[DONE]") {
      continue
    }
    chunks.push(JSON.parse(data ?? "{}") as ChatCompletionChunk)
  }
  return chunks
}

async function* asResponseStream(events: Array<Record<string, unknown>>) {
  for (const event of events) {
    await Promise.resolve()
    yield {
      data: JSON.stringify(event),
    }
  }

  await Promise.resolve()
  yield {
    data: "[DONE]",
  }
}
