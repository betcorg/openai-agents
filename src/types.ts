import {
    ChatCompletion,
    ChatCompletionCreateParamsNonStreaming,
    ChatCompletionMessageParam,
    ChatCompletionTool,
    CompletionUsage,
} from 'openai/resources';

export type ResponseChoices = string[];

export interface CompletionResult {
    choices: ResponseChoices;
    total_usage: CompletionUsage;
    completion_messages: ChatCompletionMessageParam[];
    completions: ChatCompletion[];
}

export interface HistoryOptions {
    appended_messages?: number;
    send_tool_messages?: boolean;
    remove_tool_messages?: boolean;
    ttl?: number;
    max_length?: number;
}

export type SaveHistoryOptions = Omit<
    HistoryOptions,
    'appended_messages' | 'send_tool_messages'
>;

export interface AgentCompletionParams
    extends Omit<
        ChatCompletionCreateParamsNonStreaming,
        'messages' | 'stream' | 'stream_options' | 'function_call' | 'functions'
    > {
    messages?: ChatCompletionMessageParam[] | null;
}

export interface AgentOptions extends AgentCompletionParams {
    system_instruction?: string | null;
}

export interface CreateChatCompletionOptions {
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
