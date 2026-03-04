import { Hono } from "hono"
import { streamSSE, type SSEMessage } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { forwardError } from "~/lib/error"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import {
  createResponses,
  type ResponsesApiResponse,
  type ResponsesPayload,
} from "~/services/copilot/create-responses"

export const responsesRoutes = new Hono()

responsesRoutes.post("/", async (c) => {
  try {
    await checkRateLimit(state)

    const payload = await c.req.json<ResponsesPayload>()
    if (state.manualApprove) await awaitApproval()

    const response = await createResponses(payload)
    if (isNonStreamingResponse(response)) {
      return c.json(response)
    }

    return streamSSE(c, async (stream) => {
      for await (const chunk of response) {
        await stream.writeSSE(chunk as SSEMessage)
      }
    })
  } catch (error) {
    return await forwardError(c, error)
  }
})

const isNonStreamingResponse = (
  response: Awaited<ReturnType<typeof createResponses>>,
): response is ResponsesApiResponse => !(Symbol.asyncIterator in response)
