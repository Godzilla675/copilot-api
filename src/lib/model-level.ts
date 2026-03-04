export const MODEL_LEVELS = ["low", "medium", "high", "xhigh"] as const

export type ModelLevel = (typeof MODEL_LEVELS)[number]

export const MODEL_LEVEL_VARIANTS = {
  "gpt-5.3-codex": MODEL_LEVELS,
  "claude-opus-4.6": ["low", "medium", "high"],
  "claude-opus-4.6-fast": ["low", "medium", "high"],
  "claude-sonnet-4.6": ["low", "medium", "high"],
} as const satisfies Record<string, ReadonlyArray<ModelLevel>>

export const parseModelNameWithLevel = (
  model: string,
): {
  baseModel: string
  level: ModelLevel | undefined
} => {
  const match = model.match(/^(.+)\((low|medium|high|xhigh)\)$/)
  if (!match) {
    return {
      baseModel: model,
      level: undefined,
    }
  }

  return {
    baseModel: match[1],
    level: match[2] as ModelLevel,
  }
}

export const isCodexResponsesModel = (model: string): boolean =>
  model === "gpt-5.3-codex"

export const isClaudeThinkingModel = (model: string): boolean =>
  model === "claude-opus-4.6"
  || model === "claude-opus-4.6-fast"
  || model === "claude-sonnet-4.6"
