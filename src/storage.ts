import { ChatCompletionMessageParam } from 'openai/resources';
import { RedisClientType } from 'redis';
import { HistoryOptions, SaveHistoryOptions } from './types';
import {
    MessageValidationError,
    StorageError,
    RedisKeyValidationError,
    ValidationError,
} from './errors';

const CONFIG = {
    USER_ID_MAX_LENGTH: 64,
    DEFAULT_CHAT_MAX_LENGTH: 100,
    KEY_PREFIX: 'chat:',
    DEFAULT_USER: 'default',
};

/**
 * @class AgentStorage
 * @description Manages chat history and session metadata persistence using Redis.
 */
export class AgentStorage {
    private readonly redisClient: RedisClientType;
    public historyOptions: HistoryOptions | undefined;

    constructor(client: RedisClientType) {
        if (!client) {
            throw new ValidationError('Redis client must be provided');
        }
        this.redisClient = client;
    }

    /**
     * Validates the user ID, returning a 'default' value if undefined
     * and throwing errors for invalid formats.
     */
    private validateUserId(userId?: string): string {
        if (!userId) return CONFIG.DEFAULT_USER;

        if (typeof userId !== 'string') {
            throw new RedisKeyValidationError('User ID must be a string');
        }

        if (userId.length > CONFIG.USER_ID_MAX_LENGTH) {
            throw new RedisKeyValidationError(
                `User ID exceeds maximum length of ${CONFIG.USER_ID_MAX_LENGTH} characters`
            );
        }

        if (!/^[a-zA-Z0-9_-]+$/.test(userId)) {
            throw new RedisKeyValidationError(
                'User ID contains invalid characters. Only alphanumeric, underscore, and hyphen are allowed'
            );
        }

        return userId;
    }

    /**
     * Generates the Redis key for a given user ID.
     */
    private getRedisKey(userId?: string): string {
        return `${CONFIG.KEY_PREFIX}${this.validateUserId(userId)}`;
    }

    /**
     *  Filters out tool-related messages from the chat history.
     *
     * @param {ChatCompletionMessageParam[]} messages - The chat history messages.
     * @returns {ChatCompletionMessageParam[]} The filtered messages.
     */
    public removeToolMessages(messages: ChatCompletionMessageParam[]) {
        return messages.filter(
            (message) =>
                message.role === 'user' ||
                (message.role === 'assistant' && !message.tool_calls)
        );
    }

    /**
     * Removes tool messages that don't have a corresponding assistant or tool call ID.
     *
     * @param {ChatCompletionMessageParam[]} messages - The chat history messages.
     * @returns {ChatCompletionMessageParam[]} The filtered messages.
     */
    public removeOrphanedToolMessages(
        messages: ChatCompletionMessageParam[]
    ): ChatCompletionMessageParam[] {
        const toolCallIds: Set<string> = new Set();
        const assistantCallIds: Set<string> = new Set();

        messages.forEach((message) => {
            if ('tool_call_id' in message) {
                toolCallIds.add(message.tool_call_id);
            } else if ('tool_calls' in message) {
                if (message.tool_calls)
                    message.tool_calls.forEach((toolCall) => {
                        assistantCallIds.add(toolCall.id);
                    });
            }
        });

        return messages.filter((message) => {
            if ('tool_call_id' in message) {
                return assistantCallIds.has(message.tool_call_id);
            } else if ('tool_calls' in message) {
                if (message.tool_calls) {
                    message.tool_calls = message.tool_calls.filter((toolCall) =>
                        toolCallIds.has(toolCall.id)
                    );
                    if (!message.tool_calls.length) return false;
                }
            }
            return true;
        });
    }

    private filterMessages(
        messages: ChatCompletionMessageParam[],
        options: SaveHistoryOptions
    ) {
        let filteredMessages = [...messages];
        if (filteredMessages[0].role === 'system') filteredMessages.shift();

        if (options.remove_tool_messages) {
            filteredMessages = this.removeToolMessages(filteredMessages);
        }

        return filteredMessages;
    }

    private async calculateHistoryLength(
        redisKey: string,
        messages: ChatCompletionMessageParam[],
        options: SaveHistoryOptions
    ) {
        let savedHistoryLength: number = 0;

        try {
            savedHistoryLength = await this.redisClient.lLen(redisKey);
        } catch (error) {
            throw new StorageError(
                'Error getting history length',
                error instanceof Error ? error : undefined
            );
        }

        if (
            options.max_length &&
            savedHistoryLength + messages.length > options.max_length
        ) {
            const length =
                messages.length > options.max_length
                    ? 0
                    : options.max_length - messages.length - 1;
            return length;
        }
        return CONFIG.DEFAULT_CHAT_MAX_LENGTH - messages.length - 1;
    }

    public async saveChatHistory(
        messages: ChatCompletionMessageParam[],
        userId?: string,
        options: SaveHistoryOptions = {}
    ) {
        try {
            const redisKey = this.getRedisKey(userId);

            const filteredMessages = this.filterMessages(messages, options);

            const length = await this.calculateHistoryLength(
                redisKey,
                filteredMessages,
                options
            );
            const multi = this.redisClient.multi();

            multi.lTrim(redisKey, 0, length);

            try {
                for (const message of filteredMessages) {
                    multi.lPush(redisKey, JSON.stringify(message));
                }
            } catch (error) {
                throw new MessageValidationError(
                    `Invalid message in storage: ${
                        error instanceof Error ? error.message : 'Unknown error'
                    }`,
                    error instanceof Error ? error : undefined
                );
            }

            if (options.ttl) multi.expire(redisKey, options.ttl);

            await multi.exec();
        } catch (error) {
            if (
                error instanceof MessageValidationError ||
                error instanceof RedisKeyValidationError
            )
                throw error;

            throw new StorageError(
                `Failed to save chat history: ${
                    error instanceof Error ? error.message : 'Unknown error'
                }`,
                error instanceof Error ? error : undefined
            );
        }
    }

    /**
     * Retrieves the chat history from Redis.
     *
     * @param {string} userId - The user ID.
     * @param {HistoryOptions} options - Options for retrieving the history.
     * @returns {Promise<ChatCompletionMessageParam[]>} The retrieved chat history.
     * @throws {StorageError} If retrieving the chat history fails.
     * @throws {MessageValidationError} If a message is invalid.
     * @throws {RedisKeyValidationError} If the user ID is invalid.
     */
    public async getChatHistory(userId: string, options: HistoryOptions = {}) {
        try {
            const key = this.getRedisKey(userId);

            const {
                appended_messages,
                remove_tool_messages,
                send_tool_messages,
                max_length,
            } = options;

            if (appended_messages === 0) return [];

            const messages = await this.redisClient.lRange(
                key,
                0,
                appended_messages ? appended_messages - 1 : -1
            );

            if (!messages.length) return [];

            let parsedMessages: ChatCompletionMessageParam[] = [];
            try {
                parsedMessages = messages
                    .map((message) => {
                        return JSON.parse(message);
                    })
                    .reverse() as ChatCompletionMessageParam[];
            } catch (error) {
                throw new MessageValidationError(
                    `Invalid message in storage: ${
                        error instanceof Error ? error.message : 'Unknown error'
                    }`,
                    error instanceof Error ? error : undefined
                );
            }

            if (remove_tool_messages || send_tool_messages === false)
                return this.removeToolMessages(parsedMessages);

            if (max_length)
                return this.removeOrphanedToolMessages(parsedMessages);

            return parsedMessages;
        } catch (error) {
            if (
                error instanceof MessageValidationError ||
                error instanceof RedisKeyValidationError
            )
                throw error;

            throw new StorageError(
                `Failed to retrieve stored messages: ${
                    error instanceof Error ? error.message : 'Unknown error'
                }`,
                error instanceof Error ? error : undefined
            );
        }
    }

    /**
     * Deletes the chat history from Redis for a given user ID.
     *
     * @param {string} userId - The user ID.
     * @returns {Promise<number>}
     * @throws {StorageError} If deleting the chat history fails.
     * @throws {RedisKeyValidationError} If the user ID is invalid.
     */
    public async deleteHistory(userId: string): Promise<boolean> {
        if (!userId.trim()) {
            throw new ValidationError('User ID is required');
        }
    
        try {
            const key = this.getRedisKey(userId);
            const result = await this.redisClient.del(key);
            return result > 0; 
    
        } catch (error) {
            if (error instanceof RedisKeyValidationError) {
                throw error; 
            }
            throw new StorageError(
                `Failed to delete chat history: ${error instanceof Error 
                    ? error.message 
                    : 'Unknown error occurred'}`,
                error instanceof Error ? error : undefined
            );
        }
    }
    
}
