import { CompletionUsage } from "openai/resources";

export const getTokensUsage = (
    usage1?: CompletionUsage,
    usage2?: CompletionUsage
): CompletionUsage => {
    return {
        prompt_tokens:
            (usage1?.prompt_tokens ?? 0) + (usage2?.prompt_tokens ?? 0),
        completion_tokens:
            (usage1?.completion_tokens ?? 0) +
            (usage2?.completion_tokens ?? 0),
        total_tokens:
            (usage1?.total_tokens ?? 0) + (usage2?.total_tokens ?? 0),
    };
}