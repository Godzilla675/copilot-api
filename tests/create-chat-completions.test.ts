import { test, expect, mock } from "bun:test"

import type { ChatCompletionsPayload } from "../src/services/copilot/create-chat-completions"
import type { ResponsesPayload } from "../src/services/copilot/create-responses"

import { state } from "../src/lib/state"
import { createChatCompletions } from "../src/services/copilot/create-chat-completions"
import { createResponses } from "../src/services/copilot/create-responses"

// Mock state
state.copilotToken = "test-token"
state.vsCodeVersion = "1.0.0"
state.accountType = "individual"

// Helper to mock fetch
const fetchMock = mock(
  (_url: string, opts: { headers: Record<string, string> }) => {
    return {
      ok: true,
      json: () => ({ id: "123", object: "chat.completion", choices: [] }),
      headers: opts.headers,
    }
  },
)
// @ts-expect-error - Mock fetch doesn't implement all fetch properties
;(globalThis as unknown as { fetch: typeof fetch }).fetch = fetchMock

test("sets X-Initiator to agent if tool/assistant present", async () => {
  const payload: ChatCompletionsPayload = {
    messages: [
      { role: "user", content: "hi" },
      { role: "tool", content: "tool call" },
    ],
    model: "gpt-test",
  }
  await createChatCompletions(payload)
  expect(fetchMock).toHaveBeenCalled()
  const headers = (
    fetchMock.mock.calls.at(-1)?.[1] as { headers: Record<string, string> }
  ).headers
  expect(headers["X-Initiator"]).toBe("agent")
})

test("sets X-Initiator to user if only user present", async () => {
  const payload: ChatCompletionsPayload = {
    messages: [
      { role: "user", content: "hi" },
      { role: "user", content: "hello again" },
    ],
    model: "gpt-test",
  }
  await createChatCompletions(payload)
  expect(fetchMock).toHaveBeenCalled()
  const headers = (
    fetchMock.mock.calls.at(-1)?.[1] as { headers: Record<string, string> }
  ).headers
  expect(headers["X-Initiator"]).toBe("user")
})

test("sets X-Initiator to agent for responses tool history", async () => {
  const payload: ResponsesPayload = {
    model: "gpt-5.4",
    input: [
      { role: "user", content: "hi" },
      { role: "tool", content: '{"ok":true}', tool_call_id: "call_123" },
    ],
  }

  await createResponses(payload)
  const headers = (
    fetchMock.mock.calls.at(-1)?.[1] as { headers: Record<string, string> }
  ).headers
  expect(headers["X-Initiator"]).toBe("agent")
})

test("enables vision header for responses image input", async () => {
  const payload: ResponsesPayload = {
    model: "gpt-5.4",
    input: [
      {
        role: "user",
        content: [
          { type: "input_text", text: "describe this" },
          {
            type: "input_image",
            image_url: "https://example.com/image.png",
          },
        ],
      },
    ],
  }

  await createResponses(payload)
  const headers = (
    fetchMock.mock.calls.at(-1)?.[1] as { headers: Record<string, string> }
  ).headers
  expect(headers["copilot-vision-request"]).toBe("true")
  expect(headers["X-Initiator"]).toBe("user")
})
