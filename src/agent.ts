import OpenAI, { ClientOptions } from 'openai';
import {
    ChatCompletionMessage,
    ChatCompletionToolMessageParam,
    ChatCompletionCreateParamsNonStreaming,
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
    AgentCompletionParams,
    AgentOptions,
    CompletionResult,
    ToolFunctions,
    ChatCompletion,
    CreateChatCompletionOptions,
    HistoryOptions,
} from './types';
import {
    importToolFunctions,
    loadToolsDirFiles,
} from './modules/tools-registry';
import { getCompletionsUsage, handleNResponses } from './utils';
import { RedisClientType } from 'redis';
import { AgentStorage } from './storage';
import { ValidationError } from './errors';

/**
 * A class that extends the OpenAI API client to manage chat completions and tool interactions.
 */
export class OpenAIAgent extends OpenAI {
    private static readonly REQUIRED_ENV_VARS = ['OPENAI_API_KEY'];

    private completionParams: AgentCompletionParams;

    private defaultHistoryMessages: ChatCompletionMessageParam[] | undefined;

    private Storage: AgentStorage | null = null;

    public static toolsDirPath: string | null = null;
    
    public system_instruction: string | undefined;
    
    public historyOptions: HistoryOptions | undefined;

    constructor(agentOptions: AgentOptions, options?: ClientOptions) {
        OpenAIAgent.validateEnvironment();
        super(options);
        if (!agentOptions.model) {
            throw new ValidationError(
                'Model is required to initialize the agent instance'
            );
        }
        this.system_instruction = agentOptions.system_instruction;
        delete agentOptions.system_instruction;
        this.defaultHistoryMessages = agentOptions.messages;
        delete agentOptions.messages;
        this.completionParams = agentOptions;
    }

    get model(): (string & {}) | ChatModel {
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

    private _handleSystemInstructionMessage(
        defaultInstruction: string | undefined,
        customInstruction: string | undefined
    ): ChatCompletionSystemMessageParam {
        let systemInstructionMessage: ChatCompletionSystemMessageParam = {
            role: 'system',
            content: '',
        };

        if (defaultInstruction && !customInstruction) {
            systemInstructionMessage = {
                role: 'system',
                content: defaultInstruction,
            };
        } else if (customInstruction) {
            systemInstructionMessage = {
                role: 'system',
                content: customInstruction,
            };
        }

        return systemInstructionMessage;
    }

    private async _getContextMessages(
        queryParams: ChatCompletionCreateParamsNonStreaming,
        historyOptions?: HistoryOptions
    ) {
        const userId = queryParams.user ? queryParams.user : 'default';

        let templateMessages: ChatCompletionMessageParam[] = [];
        if (this.defaultHistoryMessages) {
            templateMessages = this.defaultHistoryMessages;
        }

        if (this.Storage) {
            const storedMessages = await this.Storage.getStoredMessages(
                userId,
                historyOptions
            );
            templateMessages.push(...storedMessages);
        }
        return templateMessages;
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
                } catch (error) {
                    console.error(error);
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

    private async _handleToolCompletion(
        response: ChatCompletion,
        queryParams: ChatCompletionCreateParamsNonStreaming,
        newMessages: ChatCompletionMessageParam[],
        toolFunctions: ToolFunctions
    ) {
        if (!queryParams?.messages?.length) queryParams.messages = [];
        const responseMessage = response.choices[0].message;
        queryParams.messages.push(responseMessage);

        const toolMessages = await this._callChosenFunctions(
            responseMessage,
            toolFunctions
        );

        queryParams.messages.push(...toolMessages);
        newMessages.push(...toolMessages);

        const secondResponse = await this.chat.completions.create(
            queryParams as ChatCompletionCreateParamsNonStreaming
        );
        const secondResponseMessage = secondResponse.choices[0].message;

        if (!secondResponseMessage) {
            throw new Error(
                'No response message received from second tool query to OpenAI'
            );
        }

        newMessages.push(secondResponseMessage);
        if (this.Storage)
            await this.Storage.saveChatHistory(queryParams.user, newMessages);
        const responses = handleNResponses(secondResponse, queryParams);
        return {
            choices: responses,
            total_usage: getCompletionsUsage(response, secondResponse),
            completion_messages: newMessages,
            completions: [response, secondResponse],
        };
    }

    public async loadToolFuctions(toolsDirAddr: string): Promise<boolean> {
        if (!toolsDirAddr)
            throw new ValidationError('Tools directory path required.');
        await loadToolsDirFiles(toolsDirAddr);
        OpenAIAgent.toolsDirPath = toolsDirAddr;
        return true;
    }

    public async setStorage(
        client: RedisClientType,
        options?: { history: HistoryOptions }
    ): Promise<boolean> {
        if (!client)
            throw new ValidationError('Instance of RedisClientType required.');
        if (options?.history) this.historyOptions = options.history;
        this.Storage = new AgentStorage(client);
        return true;
    }

    public async deleteChatHistory(userId: string): Promise<boolean> {
        if (!this.Storage) {
            throw new ValidationError('Agent storage is not initalized.');
        }
        await this.Storage.deleteHistory(userId);
        return true;
    }

    public async getChatHistory(
        userId: string,
        options?: HistoryOptions
    ): Promise<ChatCompletionMessageParam[]> {
            if (this.Storage) {
                const messages = await this.Storage.getStoredMessages(
                    userId,
                    options
                );
                return messages;
            }
            return [];
    }

    public async createChatCompletion(
        options: CreateChatCompletionOptions
    ): Promise<CompletionResult> {
        const queryParams: ChatCompletionCreateParamsNonStreaming = {
            ...this.completionParams,
            ...(options.custom_params as ChatCompletionCreateParamsNonStreaming),
        };

        const historyOptions = {
            ...this.historyOptions,
            ...options.history,
        };

        if (
            this.Storage &&
            options.tool_choices &&
            this.historyOptions?.appended_messages
        )
            historyOptions.remove_tool_messages = true;

        const storedMessages =
            await this._getContextMessages(queryParams, historyOptions);

        const systemImstructionMessage = this._handleSystemInstructionMessage(
            this.system_instruction,
            options.system_instruction
        );

        if (systemImstructionMessage.content) {
            // Overwrites the default instruction if there is a new instruction in the current request
            if (storedMessages[0]?.role === 'system') storedMessages.shift();
            storedMessages.unshift(systemImstructionMessage);
        }

        const newMessages: ChatCompletionMessageParam[] = [
            { role: 'user', content: options.message },
        ];

        storedMessages.push(...newMessages);
        queryParams.messages = storedMessages;

        try {
            let toolFunctions: ToolFunctions | undefined;
            if (options.tool_choices?.length) {
                const toolChoices = await importToolFunctions(
                    options.tool_choices
                );
                queryParams.tools = toolChoices.toolChoices;
                toolFunctions = toolChoices.toolFunctions;
            }

            const response = await this.chat.completions.create(queryParams);
            const responseMessage = response.choices[0].message;

            if (!responseMessage) {
                throw new Error('No response message received from OpenAI');
            }
            newMessages.push(responseMessage);

            if (responseMessage.tool_calls && toolFunctions) {
                return await this._handleToolCompletion(
                    response,
                    queryParams,
                    newMessages,
                    toolFunctions
                );
            } else {
                if (this.Storage)
                    await this.Storage.saveChatHistory(
                        queryParams.user,
                        newMessages
                    );
                const responses = handleNResponses(response, queryParams);
                return {
                    choices: responses,
                    total_usage: getCompletionsUsage(response),
                    completion_messages: newMessages,
                    completions: [response],
                };
            }
        } catch (error) {
            const errorMessage =
                error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Chat completion failed: ${errorMessage}`);
        }
    }
}
