import OpenAI, { ClientOptions } from 'openai';
import { RedisClientType } from 'redis';
import {
    AgentCompletionParams,
    AgentOptions,
    CompletionResult,
    ToolFunctions,
    CreateChatCompletionOptions,
    HistoryOptions,
} from './types';
import {
    ChatCompletionError,
    DirectoryAccessError,
    FileImportError,
    FileReadError,
    FunctionCallError,
    InvalidToolError,
    MessageValidationError,
    RedisConnectionError,
    StorageError,
    ToolCompletionError,
    ToolNotFoundError,
    RedisKeyValidationError,
    ValidationError,
} from './errors';
import {
    importToolFunctions,
    loadToolsDirFunctions,
    ToolsRegistry,
} from './modules/tools-registry';
import { getCompletionsUsage, handleNResponses } from './utils';
import { AgentStorage } from './storage';
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
    ChatCompletion,
} from 'openai/resources';

/**
 * @class OpenAIAgent
 * @description Extends the OpenAI API client to manage chat completions, tool interactions, and persistent storage of conversation history.  Provides methods for creating chat completions, managing tools, and interacting with a Redis storage.
 */
export class OpenAIAgent extends OpenAI {
    private static readonly REQUIRED_ENV_VARS = ['OPENAI_API_KEY'];

    private completionParams: AgentCompletionParams;

    private defaultHistoryMessages: ChatCompletionMessageParam[] | null = null;

    public storage: AgentStorage | null = null;

    public system_instruction: string | null = null;

    /**
     * @constructor
     * @param {AgentOptions} agentOptions - Options for configuring the agent, including the model, system instruction, initial messages template, etc.
     * @param {ClientOptions} [options] - Optional OpenAI client options.
     * @throws {ValidationError} If the model is not specified in the agent options.
     */
    constructor(agentOptions: AgentOptions, options?: ClientOptions) {
        OpenAIAgent.validateEnvironment();
        super(options);
        if (!agentOptions.model) {
            throw new ValidationError(
                'Model is required to initialize the agent instance'
            );
        }

        if (agentOptions.system_instruction)
            this.system_instruction = agentOptions.system_instruction;
        delete agentOptions.system_instruction;

        if (agentOptions.messages)
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
     */
    private static validateEnvironment(): void {
        const missingVars = OpenAIAgent.REQUIRED_ENV_VARS.filter(
            (varName) => !process.env[varName]
        );
        if (missingVars.length > 0) {
            throw new ValidationError(
                `Missing required environment variables: ${missingVars.join(
                    ', '
                )}`
            );
        }
    }

    /**
     * Determines the system instruction message to use based on default and custom instructions.
     */
    private handleSystemInstructionMessage(
        defaultInstruction: string | undefined | null,
        customInstruction: string | undefined
    ): ChatCompletionSystemMessageParam {
        const systemInstructionMessage: ChatCompletionSystemMessageParam = {
            role: 'system',
            content: '',
        };

        if (defaultInstruction && !customInstruction) {
            systemInstructionMessage.content = defaultInstruction;
        } else if (customInstruction) {
            systemInstructionMessage.content = customInstruction;
        }

        return systemInstructionMessage;
    }

    /**
     * Retrieves context messages from the default history and/or from persistent storage.
     */
    private async handleContextMessages(
        queryParams: ChatCompletionCreateParamsNonStreaming,
        historyOptions: HistoryOptions
    ): Promise<ChatCompletionMessageParam[]> {
        const userId = queryParams.user ? queryParams.user : 'default';

        let contextMessages: ChatCompletionMessageParam[] = [];
        if (this.defaultHistoryMessages) {
            contextMessages = this.defaultHistoryMessages;
        }

        if (this.storage) {
            const storedMessages = await this.storage.getChatHistory(
                userId,
                historyOptions
            );
            contextMessages.push(...storedMessages);
        }
        console.log('Context length:', contextMessages.length);
        return contextMessages;
    }

    /**
     * Executes the functions called by the model and returns their responses.
     */
    private async callChosenFunctions(
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
                toolMessages.push({
                    tool_call_id: id,
                    role: 'tool',
                    content: JSON.stringify({
                        error:
                            error instanceof Error
                                ? error.message
                                : 'Unknown error',
                    }),
                });
                throw new FunctionCallError(
                    name,
                    error instanceof Error ? error.message : 'Unknown error'
                );
            }
        }

        return toolMessages;
    }

    /**
     * Handles the process of calling tools based on the model's response
     * and making a subsequent API call with the tool responses.
     */
    private async handleToolCompletion(toolCompletionOpts: {
        response: ChatCompletion;
        queryParams: ChatCompletionCreateParamsNonStreaming;
        newMessages: ChatCompletionMessageParam[];
        toolFunctions: ToolFunctions;
        historyOptions: HistoryOptions;
    }): Promise<CompletionResult> {
        const {
            response,
            queryParams,
            newMessages,
            toolFunctions,
            historyOptions,
        } = toolCompletionOpts;
        if (!queryParams?.messages?.length) queryParams.messages = [];
        const responseMessage = response.choices[0].message;
        queryParams.messages.push(responseMessage);

        try {
            const toolMessages = await this.callChosenFunctions(
                responseMessage,
                toolFunctions
            );
            queryParams.messages.push(...toolMessages);
            newMessages.push(...toolMessages);

            const secondResponse = await this.chat.completions.create(
                queryParams as ChatCompletionCreateParamsNonStreaming
            );

            const secondResponseMessage = secondResponse.choices[0].message;

            newMessages.push(secondResponseMessage);
            if (this.storage) {
                await this.storage.saveChatHistory(
                    newMessages,
                    queryParams.user,
                    historyOptions
                );
            }

            const responses = handleNResponses(secondResponse, queryParams);
            return {
                choices: responses,
                total_usage: getCompletionsUsage(response, secondResponse),
                completion_messages: newMessages,
                completions: [response, secondResponse],
            };
        } catch (error) {
            if (error instanceof FunctionCallError) throw error;
            throw new ToolCompletionError(
                queryParams,
                error instanceof Error ? error : undefined
            );
        }
    }

    /**
     * Creates a chat completion, handles tool calls (if any), and manages conversation history.
     *
     * @param {string} message - The user's message.
     * @param {CreateChatCompletionOptions} [completionOptions] - Options for the chat completion, including custom parameters, tool choices, and history management.
     * @returns {Promise<CompletionResult>} A promise that resolves to the chat completion result.
     * @throws {ChatCompletionError | StorageError | RedisConnectionError | RedisKeyValidationError | MessageValidationError | DirectoryAccessError | FileReadError | FileImportError | InvalidToolError | ToolNotFoundError | ToolCompletionError | FunctionCallError | ValidationError} If an error occurs during the completion process.
     */
    public async createChatCompletion(
        message: string,
        completionOptions: CreateChatCompletionOptions = {}
    ): Promise<CompletionResult> {
        const queryParams: ChatCompletionCreateParamsNonStreaming = {
            ...this.completionParams,
            ...(completionOptions.custom_params as ChatCompletionCreateParamsNonStreaming),
        };

        const historyOptions = {
            ...this.storage?.historyOptions,
            ...completionOptions.history,
        };

        let storedMessages: ChatCompletionMessageParam[] = [];

        try {
            storedMessages = await this.handleContextMessages(
                queryParams,
                historyOptions
            );

            if (
                this.storage &&
                completionOptions.tool_choices &&
                historyOptions?.appended_messages
            )
                storedMessages =
                    this.storage.removeOrphanedToolMessages(storedMessages);

            const systemImstructionMessage =
                this.handleSystemInstructionMessage(
                    this.system_instruction,
                    completionOptions.system_instruction
                );

            if (systemImstructionMessage.content) {
                // Overwrites the default instruction if there is a new instruction in the current request
                if (storedMessages[0]?.role === 'system')
                    storedMessages.shift();
                storedMessages.unshift(systemImstructionMessage);
            }

            const newMessages: ChatCompletionMessageParam[] = [
                { role: 'user', content: message },
            ];

            storedMessages.push(...newMessages);
            queryParams.messages = storedMessages;

            let toolFunctions: ToolFunctions | undefined;
            if (completionOptions.tool_choices?.length) {
                const toolChoices = await importToolFunctions(
                    completionOptions.tool_choices
                );
                queryParams.tools = toolChoices.toolChoices;
                toolFunctions = toolChoices.toolFunctions;
            }

            const response = await this.chat.completions.create(queryParams);
            const responseMessage = response.choices[0].message;

            newMessages.push(responseMessage);

            if (responseMessage.tool_calls && toolFunctions) {
                return await this.handleToolCompletion({
                    response,
                    queryParams,
                    newMessages,
                    toolFunctions,
                    historyOptions,
                });
            } else {
                if (this.storage) {
                    await this.storage.saveChatHistory(
                        newMessages,
                        queryParams.user,
                        historyOptions
                    );
                }
                const responses = handleNResponses(response, queryParams);
                return {
                    choices: responses,
                    total_usage: getCompletionsUsage(response),
                    completion_messages: newMessages,
                    completions: [response],
                };
            }
        } catch (error) {
            if (
                error instanceof StorageError ||
                error instanceof RedisConnectionError ||
                error instanceof RedisKeyValidationError ||
                error instanceof MessageValidationError ||
                error instanceof DirectoryAccessError ||
                error instanceof FileReadError ||
                error instanceof FileImportError ||
                error instanceof InvalidToolError ||
                error instanceof ToolNotFoundError ||
                error instanceof ToolCompletionError ||
                error instanceof FunctionCallError ||
                error instanceof ValidationError
            )
                throw error;
            throw new ChatCompletionError(
                queryParams,
                error instanceof Error ? error : undefined
            );
        }
    }

    /**
     * Loads tool functions from the specified directory.
     *
     * @param {string} toolsDirPath - The path to the directory containing the tool functions.
     * @returns {Promise<boolean>} A promise that resolves to true if the tools are loaded successfully.
     * @throws {ValidationError} If the tools directory path is not provided or invalid.
     */
    public async loadToolFuctions(toolsDirPath: string): Promise<boolean> {
        if (!toolsDirPath)
            throw new ValidationError('Tools directory path required.');
        await loadToolsDirFunctions(toolsDirPath);
        ToolsRegistry.toolsDirPath = toolsDirPath;
        return true;
    }

    /**
     * Sets up the storage using a Redis client.
     *
     * @param {RedisClientType} client - The Redis client instance.
     * @param {{ history: HistoryOptions }} [options] - Options for configuring history storage.
     * @returns {Promise<boolean>} A promise that resolves to true if the storage is set up successfully.
     * @throws {RedisConnectionError} If the Redis client is not provided.
     */
    public async setStorage(
        client: RedisClientType,
        options?: { history: HistoryOptions }
    ): Promise<boolean> {
        if (!client)
            throw new RedisConnectionError('Instance of Redis is required.');

        this.storage = new AgentStorage(client);
        if (options?.history) this.storage.historyOptions = options.history;
        return true;
    }

    /**
     * Deletes the chat history for a given user.
     *
     * @param {string} userId - The ID of the user whose history should be deleted.
     * @returns {Promise<boolean>} A promise that resolves to true if the history is deleted successfully.
     * @throws {RedisConnectionError} If the storage is not initialized.
     */
    public async deleteChatHistory(userId: string): Promise<boolean> {
        if (!this.storage) {
            throw new RedisConnectionError('Agent storage is not initalized.');
        }
        await this.storage.deleteHistory(userId);
        return true;
    }

    /**
     * Retrieves the chat history for a given user.
     *
     * @param {string} userId - The ID of the user whose history should be retrieved.
     * @param {HistoryOptions} [options] - Options for retrieving history.
     * @returns {Promise<ChatCompletionMessageParam[]>} A promise that resolves to an array of chat messages.
     * @throws {RedisConnectionError} If the storage is not initialized.
     */
    public async getChatHistory(
        userId: string,
    ): Promise<ChatCompletionMessageParam[]> {
        if (!this.storage)
            throw new RedisConnectionError('Agent storage is not initialized');
        const messages = await this.storage.getChatHistory(userId);
        return messages;
    }
}
