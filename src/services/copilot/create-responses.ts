import consola from "consola"
import { events } from "fetch-event-stream"

import { copilotHeaders, copilotBaseUrl } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { type ModelLevel } from "~/lib/model-level"
import { state } from "~/lib/state"

export const createResponses = async (payload: ResponsesPayload) => {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  const response = await fetch(`${copilotBaseUrl(state)}/v1/responses`, {
    method: "POST",
    headers: copilotHeaders(state),
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
  input: string | Array<ResponseInputMessage>
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
  } | null
}

export interface ResponseInputMessage {
  role: "user" | "assistant" | "system" | "tool" | "developer"
  content: string | Array<ResponseInputContentPart>
}

export interface ResponseInputContentPart {
  type: "input_text" | "input_image"
  text?: string
  image_url?: string
  detail?: "low" | "high" | "auto"
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
  }
}

export type ResponsesOutputItem = ResponsesOutputMessage | ResponsesFunctionCall

export interface ResponsesOutputMessage {
  type: "message"
  role: "assistant" | "user" | "system" | "tool"
  content: string | Array<ResponsesOutputContentPart>
}

export type ResponsesOutputContentPart =
  | {
      type: "output_text"
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
  arguments: string
  call_id?: string
  id?: string
}
