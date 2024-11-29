import {
    ChatCompletion,
    ChatCompletionCreateParamsNonStreaming,
    CompletionUsage,
} from 'openai/resources';
import { ResponseChoices } from './types';

/**
 * Calculates the sum of prompt tokens, completion tokens, and total tokens from multiple `CompletionUsage` objects.
 */
export const getTokensSum = (...usages: CompletionUsage[]): CompletionUsage => {
    const initialUsage: CompletionUsage = {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };

    return usages.reduce((accumulator, currentUsage) => {
        return {
            prompt_tokens:
                accumulator.prompt_tokens + (currentUsage?.prompt_tokens ?? 0),
            completion_tokens:
                accumulator.completion_tokens +
                (currentUsage?.completion_tokens ?? 0),
            total_tokens:
                accumulator.total_tokens + (currentUsage?.total_tokens ?? 0),
        };
    }, initialUsage);
};

/**
 * Extracts and aggregates the `usage` information from multiple `ChatCompletion` objects.
 *
 * @param {...ChatCompletion} completions - One or more `ChatCompletion` objects.
 * @returns {CompletionUsage} A `CompletionUsage` object representing the aggregated usage data.
 * Returns an object with all properties set to 0 if no completions are provided or if none of them have a usage property.
 */
export const getCompletionsUsage = (
    ...completions: ChatCompletion[]
): CompletionUsage => {
    const usages: CompletionUsage[] = [];
    for (const completion of completions) {
        if (completion.usage) {
            usages.push(completion.usage);
        }
    }
    return getTokensSum(...usages);
};

export const handleNResponses = (
    response: ChatCompletion,
    queryParams: ChatCompletionCreateParamsNonStreaming
): ResponseChoices => {
    let responses: ResponseChoices = [];
    const responseMessage = response.choices[0].message;
    if (queryParams.n) {
        for (const choice of response.choices) {
            if (choice.message.content) responses.push(choice.message.content);
        }
    } else {
        responses = [responseMessage.content ?? 'Response not received.'];
    }
    return responses;
};
