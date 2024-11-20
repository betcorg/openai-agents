import {
    ChatCompletionMessage,
    ChatCompletionToolMessageParam,
    ChatCompletionCreateParamsNonStreaming,
    ChatCompletion,
    CompletionUsage,
    ChatCompletionMessageParam,
    ChatModel,
    ChatCompletionAudioParam,
    ChatCompletionModality,
    ChatCompletionPredictionContent,
    ResponseFormatText,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ChatCompletionToolChoiceOption,
    ChatCompletionSystemMessageParam,
    ChatCompletionTool,
    FunctionDefinition,
    FunctionParameters,
} from 'openai/resources/index.js';

export type {
    ChatCompletionMessage,
    ChatCompletionToolMessageParam,
    ChatCompletionCreateParamsNonStreaming,
    ChatCompletion,
    CompletionUsage,
    ChatCompletionMessageParam,
    ChatModel,
    ChatCompletionAudioParam,
    ChatCompletionModality,
    ChatCompletionPredictionContent,
    ResponseFormatText,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ChatCompletionToolChoiceOption,
    ChatCompletionSystemMessageParam,
    ChatCompletionTool,
    FunctionDefinition,
    FunctionParameters,
};

export type ResponseChoices = string[];

export interface CompletionResult {
    choices: ResponseChoices;
    total_usage: CompletionUsage;
    completion_messages: ChatCompletionMessageParam[];
    completions: ChatCompletion[];
}

export interface HistoryOptions {
    appended_messages?: number;
    remove_tool_messages?: boolean;
}

export interface AgentCompletionParams
    extends Omit<
        ChatCompletionCreateParamsNonStreaming,
        'messages' | 'stream' | 'stream_options' | 'function_call' | 'functions'
    > {
    messages?: ChatCompletionMessageParam[];
}

export interface AgentOptions extends AgentCompletionParams {
    system_instruction?: string;
}

export interface CreateChatCompletionOptions {
    message: string;
    system_instruction?: string;
    tool_choices?: string[];
    custom_params?: Partial<AgentCompletionParams>;
    history?: HistoryOptions;
}

export type ToolFunction = (
    args: object
) => Promise<string> | string | undefined;

export interface ToolFunctions {
    [key: string]: (args: object) => Promise<string> | string | undefined;
}

export interface AgentTools {
    toolDefinitions: ChatCompletionTool[];
    toolFunctions: ToolFunctions;
}

export interface ToolChoices {
    toolChoices: ChatCompletionTool[];
    toolFunctions: ToolFunctions;
}
