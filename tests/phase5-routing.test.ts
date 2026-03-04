import { describe, expect, test } from "bun:test"

import type { Model } from "~/services/copilot/get-models"

import {
  MODEL_LEVEL_VARIANTS,
  parseModelNameWithLevel,
} from "~/lib/model-level"
import {
  translateChatCompletionsToResponses,
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

  test("maps codex suffix level to reasoning_effort", () => {
    const payload = normalizeChatCompletionsPayloadModel({
      model: "gpt-5.3-codex(xhigh)",
      messages: [{ role: "user", content: "hi" }],
    })

    expect(payload.model).toBe("gpt-5.3-codex")
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
      model: "gpt-5.3-codex",
      messages: [{ role: "user", content: "Hello" }],
      max_tokens: 128,
      reasoning_effort: "medium",
    })

    expect(translated).toMatchObject({
      model: "gpt-5.3-codex",
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
})

describe("model listing expansion", () => {
  test("includes required level-suffixed variants", () => {
    const models = expandModelList([
      makeModel("gpt-5.3-codex"),
      makeModel("claude-opus-4.6"),
      makeModel("claude-opus-4.6-fast"),
      makeModel("claude-sonnet-4.6"),
      makeModel("gpt-4.1"),
    ])
    const ids = models.map((model) => model.id)

    expect(ids).toContain("gpt-5.3-codex")
    for (const level of MODEL_LEVEL_VARIANTS["gpt-5.3-codex"]) {
      expect(ids).toContain(`gpt-5.3-codex(${level})`)
    }
    for (const level of MODEL_LEVEL_VARIANTS["claude-opus-4.6"]) {
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
