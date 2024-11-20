import { RedisClientType } from 'redis';
import { ChatCompletionMessageParam, HistoryOptions } from './types';

export class AgentStorage {
    private redisClient: RedisClientType | null = null;

    constructor(client: RedisClientType) {
        this.redisClient = client;
    }

    public async saveChatHistory(
        userId: string | undefined,
        messages: ChatCompletionMessageParam[]
    ): Promise<ChatCompletionMessageParam[]> {
        if (!this.redisClient) {
            throw new Error('Redis client not initialized.');
        }
        const id = userId ? userId : 'default';
        if (messages[0]?.role === 'system') messages.shift();

        try {
            for (const message of messages) {
                await this.redisClient.lPush(
                    `user:${id}`,
                    JSON.stringify(message)
                );
            }
            return await this.getStoredMessages(id);
        } catch (error) {
            throw new Error(
                `Error saving chat history of user: ${userId}: ${
                    error instanceof Error ? error.message : 'Unknown error'
                }`
            );
        }
    }

    public async deleteHistory(userId: string): Promise<number> {
        if (!this.redisClient) {
            throw new Error('Redis client not initialized.');
        }

        try {
            return await this.redisClient.del(`user:${userId}`);
        } catch (error) {
            throw new Error(
                `Error deleting chat history of user: ${userId}: ${
                    error instanceof Error ? error.message : 'Unknown error'
                }`
            );
        }
    }

    public async getStoredMessages(
        userId: string,
        options: HistoryOptions = {}
    ): Promise<ChatCompletionMessageParam[]> {
        if (!this.redisClient) {
            throw new Error('Redis client not initialized.');
        }
        const { appended_messages, remove_tool_messages } = options;
        let messages: string[];

        if (appended_messages === 0) {
            return [];
        } else if (appended_messages) {
            messages = await this.redisClient.lRange(
                `user:${userId}`,
                0,
                appended_messages - 1
            );
        } else {
            messages = await this.redisClient.lRange(`user:${userId}`, 0, -1);
        }

        if (!messages || messages.length === 0) return [];

        const parsedMessages = messages
            .map((message) => JSON.parse(message))
            .reverse() as ChatCompletionMessageParam[];

        if (remove_tool_messages) {
            return parsedMessages.filter(
                (message: ChatCompletionMessageParam) =>
                    message.role === 'user' ||
                    (message.role === 'assistant' && !message.tool_calls)
            );
        }
        return parsedMessages;
    }
}
