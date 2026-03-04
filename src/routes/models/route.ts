import { Hono } from "hono"

import type { Model } from "~/services/copilot/get-models"

import { forwardError } from "~/lib/error"
import { MODEL_LEVEL_VARIANTS } from "~/lib/model-level"
import { state } from "~/lib/state"
import { cacheModels } from "~/lib/utils"

export const modelRoutes = new Hono()

modelRoutes.get("/", async (c) => {
  try {
    if (!state.models) {
      // This should be handled by startup logic, but as a fallback.
      await cacheModels()
    }

    const models = expandModelList(state.models?.data ?? [])

    return c.json({
      object: "list",
      data: models,
      has_more: false,
    })
  } catch (error) {
    return await forwardError(c, error)
  }
})

export function expandModelList(models: Array<Model>) {
  return models.flatMap((model) => {
    const expanded = [toModelItem(model, model.id)]
    const levels =
      model.id in MODEL_LEVEL_VARIANTS ?
        MODEL_LEVEL_VARIANTS[model.id as keyof typeof MODEL_LEVEL_VARIANTS]
      : undefined
    if (!levels) {
      return expanded
    }

    for (const level of levels) {
      expanded.push(toModelItem(model, `${model.id}(${level})`))
    }

    return expanded
  })
}

function toModelItem(model: Model, id: string) {
  return {
    id,
    object: "model",
    type: "model",
    created: 0,
    created_at: new Date(0).toISOString(),
    owned_by: model.vendor,
    display_name: model.name,
  }
}
