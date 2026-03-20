import consola from "consola"
import { events } from "fetch-event-stream"

import { copilotHeaders, copilotBaseUrl } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { type ModelLevel } from "~/lib/model-level"
import { state } from "~/lib/state"

export const createResponses = async (payload: ResponsesPayload) => {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  const headers: Record<string, string> = {
    ...copilotHeaders(state, hasVisionInput(payload.input)),
    "X-Initiator": hasAgentInput(payload.input) ? "agent" : "user",
  }

  const response = await fetch(`${copilotBaseUrl(state)}/v1/responses`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    consola.error("Failed to create response", response)
    throw new HTTPError("Failed to create response", response)
  }

  if (payload.stream) {
    return events(response)
  }

  return (await response.json()) as ResponsesApiResponse
}

export interface ResponsesPayload {
  model: string
  input: string | Array<ResponseInputItem>
  stream?: boolean | null
  temperature?: number | null
  top_p?: number | null
  max_output_tokens?: number | null
  stop?: string | Array<string> | null
  tools?: Array<unknown> | null
  tool_choice?: unknown
  user?: string | null
  reasoning_effort?: ModelLevel | null
  reasoning?: {
    effort?: ModelLevel
    summary?: string
  } | null
  include?: Array<string> | null
}

export type ResponseInputItem = ResponseInputMessage | ResponseInputReasoning

export interface ResponseInputMessage {
  role: "user" | "assistant" | "system" | "tool" | "developer"
  content: string | Array<ResponseInputContentPart>
  name?: string
  tool_call_id?: string
  tool_calls?: Array<ResponseInputToolCall>
}

export interface ResponseInputReasoning {
  type: "reasoning"
  encrypted_content: string
  id?: string
  summary?: Array<{
    type: "summary_text"
    text: string
  }>
}

export interface ResponseInputContentPart {
  type: "input_text" | "input_image"
  text?: string
  image_url?: string
  detail?: "low" | "high" | "auto"
}

export interface ResponseInputToolCall {
  id: string
  type: "function"
  function: {
    name: string
    arguments: string
  }
}

export interface ResponsesApiResponse {
  id: string
  object: string
  created_at?: number
  model: string
  output?: Array<ResponsesOutputItem>
  output_text?: string
  usage?: {
    input_tokens?: number
    output_tokens?: number
    total_tokens?: number
    input_tokens_details?: {
      cached_tokens?: number
    }
  }
}

export type ResponsesOutputItem =
  | ResponsesOutputMessage
  | ResponsesFunctionCall
  | ResponsesReasoningItem

export interface ResponsesOutputMessage {
  type: "message"
  role: "assistant" | "user" | "system" | "tool"
  content: string | Array<ResponsesOutputContentPart>
  id?: string
  status?: string
  object?: string
}

export type ResponsesOutputContentPart =
  | {
      type: "output_text"
      text: string
    }
  | {
      type: "text"
      text: string
    }
  | ResponsesFunctionCall
  | {
      type: string
      [key: string]: unknown
    }

export interface ResponsesFunctionCall {
  type: "function_call"
  name: string
  arguments?: string
  call_id?: string
  id?: string
  object?: string
  status?: string
}

export interface ResponsesReasoningItem {
  type: "reasoning"
  id?: string
  encrypted_content?: string
  summary?: Array<{
    type: string
    text?: string
  }>
}

function hasVisionInput(input: ResponsesPayload["input"]): boolean {
  if (!Array.isArray(input)) {
    return false
  }

  return input.some(
    (message) =>
      isResponseInputMessage(message)
      && Array.isArray(message.content)
      && message.content.some((part) => part.type === "input_image"),
  )
}

function hasAgentInput(input: ResponsesPayload["input"]): boolean {
  if (!Array.isArray(input)) {
    return false
  }

  return input.some(
    (message) =>
      isResponseInputMessage(message)
      && ["assistant", "tool"].includes(message.role),
  )
}

function isResponseInputMessage(
  value: ResponseInputItem,
): value is ResponseInputMessage {
  return "role" in value
}
