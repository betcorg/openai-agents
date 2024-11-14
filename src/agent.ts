import OpenAI, { ClientOptions } from 'openai';
import {
    ChatCompletionMessage,
    ChatCompletionToolMessageParam,
    ChatCompletionCreateParamsNonStreaming,
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
} from 'openai/resources/index.js';
import {
    importToolFunctions,
    loadToolFunctions,
    ToolFunctions,
} from './modules/toolsLoader';
import { getTokensUsage } from './utils';

export interface CompletionResult {
    history: ChatCompletionMessageParam[];
    textResponse: string;
    usage: CompletionUsage;
}

interface AgentOptions
    extends Omit<ChatCompletionCreateParamsNonStreaming, 'messages'> {
    messages?: ChatCompletionMessageParam[];
    systemInstruction?: string;
}

type CompletionParams = Omit<AgentOptions, 'systemInstruction'>;

/**
 * A class that extends the OpenAI API client to manage chat completions and tool interactions.
 */
export class OpenAIAgent extends OpenAI {
    /** Required environment variables. */
    private static readonly REQUIRED_ENV_VARS = ['OPENAI_API_KEY'];
    /** Parameters for chat completion requests. */
    private completionParams: CompletionParams;
    /** Path to the directory containing tool function definitions. */
    public static toolsDirPath: string | null = null;

    public systemInstruction: string | undefined;

    constructor(agentOptions: AgentOptions, options?: ClientOptions) {
        OpenAIAgent.validateEnvironment();
        super(options);

        if (!agentOptions.model) {
            throw new Error(
                'Model is required to initialize the agent instance'
            );
        }

        this.systemInstruction = agentOptions.systemInstruction;
        delete agentOptions.systemInstruction;
        this.completionParams = agentOptions;
    }

    get model(): (string & {}) | ChatModel | undefined {
        return this.completionParams.model;
    }
    set model(value: (string & {}) | ChatModel) {
        this.completionParams.model = value;
    }

    get temperature(): number | null | undefined {
        return this.completionParams.temperature;
    }
    set temperature(value: number | null | undefined) {
        this.completionParams.temperature = value;
    }

    get top_p(): number | null | undefined {
        return this.completionParams.top_p;
    }
    set top_p(value: number | null | undefined) {
        this.completionParams.top_p = value;
    }

    get max_completion_tokens(): number | null | undefined {
        return this.completionParams.max_completion_tokens;
    }
    set max_completion_tokens(value: number | null | undefined) {
        this.completionParams.max_completion_tokens = value;
    }

    get max_tokens(): number | null | undefined {
        return this.completionParams.max_tokens;
    }
    set max_tokens(value: number | null | undefined) {
        this.completionParams.max_tokens = value;
    }

    get n(): number | null | undefined {
        return this.completionParams.n;
    }
    set n(value: number | null | undefined) {
        this.completionParams.n = value;
    }

    get frequency_penalty(): number | null | undefined {
        return this.completionParams.frequency_penalty;
    }
    set frequency_penalty(value: number | null | undefined) {
        this.completionParams.frequency_penalty = value;
    }

    get presence_penalty(): number | null | undefined {
        return this.completionParams.presence_penalty;
    }

    set presence_penalty(value: number | null | undefined) {
        this.completionParams.presence_penalty = value;
    }

    get history(): ChatCompletionMessageParam[] | undefined {
        return this.completionParams.messages;
    }
    set history(value: ChatCompletionMessageParam[]) {
        this.completionParams.messages = value;
    }

    get tool_choice(): ChatCompletionToolChoiceOption | undefined {
        return this.completionParams.tool_choice;
    }
    set tool_choice(value: ChatCompletionToolChoiceOption | undefined) {
        this.completionParams.tool_choice = value;
    }

    get parallel_tool_calls(): boolean | undefined {
        return this.completionParams.parallel_tool_calls;
    }
    set parallel_tool_calls(value: boolean | undefined) {
        this.completionParams.parallel_tool_calls = value;
    }

    get audioParams(): ChatCompletionAudioParam | null | undefined {
        return this.completionParams.audio;
    }
    set audioParams(value: ChatCompletionAudioParam | null | undefined) {
        this.completionParams.audio = value;
    }

    get response_format():
        | ResponseFormatText
        | ResponseFormatJSONObject
        | ResponseFormatJSONSchema
        | undefined {
        return this.completionParams.response_format;
    }
    set response_format(
        value:
            | ResponseFormatText
            | ResponseFormatJSONObject
            | ResponseFormatJSONSchema
            | undefined
    ) {
        this.completionParams.response_format = value;
    }

    get logit_bias(): Record<string, number> | null | undefined {
        return this.completionParams.logit_bias;
    }
    set logit_bias(value: Record<string, number> | null | undefined) {
        this.completionParams.logit_bias = value;
    }

    get logprobs(): boolean | null | undefined {
        return this.completionParams.logprobs;
    }
    set logprobs(value: boolean | null | undefined) {
        this.completionParams.logprobs = value;
    }

    get top_logprobs(): number | null | undefined {
        return this.completionParams.top_logprobs;
    }
    set top_logprobs(value: number | null | undefined) {
        this.completionParams.top_logprobs = value;
    }

    get metadata(): Record<string, string> | null | undefined {
        return this.completionParams.metadata;
    }

    set metadata(value: Record<string, string> | null | undefined) {
        this.completionParams.metadata = value;
    }

    get stop(): string | null | string[] | undefined {
        return this.completionParams.stop;
    }
    set stop(value: string | null | string[] | undefined) {
        this.completionParams.stop = value;
    }

    get modalities(): ChatCompletionModality[] | null | undefined {
        return this.completionParams.modalities;
    }
    set modalities(value: ChatCompletionModality[] | null | undefined) {
        this.completionParams.modalities = value;
    }

    get prediction(): ChatCompletionPredictionContent | null | undefined {
        return this.completionParams.prediction;
    }
    set prediction(value: ChatCompletionPredictionContent | null | undefined) {
        this.completionParams.prediction = value;
    }

    get seed(): number | null | undefined {
        return this.completionParams.seed;
    }
    set seed(value: number | null | undefined) {
        this.completionParams.seed = value;
    }

    get service_tier(): 'auto' | 'default' | null | undefined {
        return this.completionParams.service_tier;
    }
    set service_tier(value: 'auto' | 'default' | null | undefined) {
        this.completionParams.service_tier = value;
    }

    get store(): boolean | null | undefined {
        return this.completionParams.store;
    }
    set store(value: boolean | null | undefined) {
        this.completionParams.store = value;
    }

    /**
     * Validates that required environment variables are set.
     * @throws {Error} If any required environment variables are missing.
     */
    private static validateEnvironment(): void {
        const missingVars = OpenAIAgent.REQUIRED_ENV_VARS.filter(
            (varName) => !process.env[varName]
        );
        if (missingVars.length > 0) {
            throw new Error(
                `Missing required environment variables: ${missingVars.join(
                    ', '
                )}`
            );
        }
    }

    public async loadToolFuctions(toolsDirAddr: string): Promise<boolean> {
        const tools = await loadToolFunctions(toolsDirAddr);
        if (!tools) return false;
        OpenAIAgent.toolsDirPath = toolsDirAddr;
        return true;
    }

    private async _callChosenFunctions(
        responseMessage: ChatCompletionMessage,
        functions: ToolFunctions
    ): Promise<ChatCompletionToolMessageParam[]> {
        if (!responseMessage.tool_calls?.length) {
            throw new Error('No tool calls found in the response message');
        }

        const toolMessages: ChatCompletionToolMessageParam[] = [];

        for (const tool of responseMessage.tool_calls) {
            const {
                id,
                function: { name, arguments: args },
            } = tool;

            try {
                const currentFunction = functions[name];
                if (!currentFunction) {
                    throw new Error(`Function '${name}' not found`);
                }

                let parsedArgs;
                try {
                    parsedArgs = JSON.parse(args);
                } catch (e) {
                    console.log(e);
                    throw new Error(
                        `Invalid arguments format for function '${name}': ${args}`
                    );
                }

                const functionResponse = await Promise.resolve(
                    currentFunction(parsedArgs)
                );

                if (functionResponse === undefined) {
                    throw new Error(`Function '${name}' returned no response`);
                }

                toolMessages.push({
                    tool_call_id: id,
                    role: 'tool',
                    content: JSON.stringify(functionResponse),
                });
            } catch (error) {
                const errorMessage =
                    error instanceof Error ? error.message : 'Unknown error';
                console.error(
                    `Error calling function '${name}':`,
                    errorMessage
                );

                toolMessages.push({
                    tool_call_id: id,
                    role: 'tool',
                    content: JSON.stringify({ error: errorMessage }),
                });
            }
        }

        return toolMessages;
    }


    public async createChatCompletion(options: {
        message: string;
        systemInstruction?: string;
        toolNames?: string[];
        customParams?: Partial<ChatCompletionCreateParamsNonStreaming>;
    }): Promise<CompletionResult> {
        const messages: ChatCompletionMessageParam[] = [
            { role: 'user', content: options.message },
        ];

        if (this.systemInstruction && !options.systemInstruction) {
            messages.unshift({
                role: 'system',
                content: this.systemInstruction,
            });
        } else if (options.systemInstruction) {
            messages.unshift({
                role: 'system',
                content: options.systemInstruction,
            });
        }

        this.history = messages;

        const currentParams = {
            ...this.completionParams,
            ...options.customParams,
        };
        try {
            let toolFunctions: ToolFunctions | undefined;
            if (options.toolNames?.length) {
                const toolChoices = await importToolFunctions(
                    options.toolNames
                );
                currentParams.tools = toolChoices.toolChoices;
                toolFunctions = toolChoices.toolFunctions;
            }

            console.log({ currentParams });
            const response = await this.chat.completions.create(
                currentParams as ChatCompletionCreateParamsNonStreaming
            );
            const responseMessage = response.choices[0].message;

            if (!responseMessage) {
                throw new Error('No response message received from OpenAI');
            }

            if (responseMessage.tool_calls && toolFunctions) {
                this.history.push(responseMessage);
                const toolMessages = await this._callChosenFunctions(
                    responseMessage,
                    toolFunctions
                );
                this.history.push(...toolMessages);

                const secondResponse = await this.chat.completions.create(
                    currentParams as ChatCompletionCreateParamsNonStreaming
                );

                const secondResponseMessage = secondResponse.choices[0].message;

                const usage = getTokensUsage(
                    response.usage,
                    secondResponse.usage
                );
                this.history.push(secondResponseMessage);
                return {
                    textResponse: secondResponseMessage.content ?? '',
                    usage,
                    history: this.history,
                };
            } else {
                this.history.push(responseMessage);
                return {
                    textResponse: responseMessage.content ?? '',
                    usage: getTokensUsage(response.usage),
                    history: this.history,
                };
            }
        } catch (error) {
            const errorMessage =
                error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Chat completion failed: ${errorMessage}`);
        }
    }
}
