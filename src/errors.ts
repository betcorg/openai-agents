import { ChatCompletionCreateParamsNonStreaming } from "openai/resources";

// Base Error
export abstract class BaseError extends Error {
    constructor(message: string, public readonly cause?: Error) {
        super(message);
        this.name = this.constructor.name;
        Error.captureStackTrace(this, this.constructor);
    }
}

// Validation Errors
export class ValidationError extends BaseError {
    constructor(message: string, cause?: Error) {
        super(message, cause);
    }
}

export class MessageValidationError extends ValidationError {
    constructor(message: string, cause?: Error) {
        super(message, cause);
    }
}

// File System Errors
export class FileSystemError extends BaseError {
    constructor(
        message: string,
        public readonly path: string,
        cause?: Error
    ) {
        super(`${message}: ${path}`, cause);
    }
}

export class DirectoryAccessError extends FileSystemError {
    constructor(dirPath: string, cause?: Error) {
        super('Unable to access directory', dirPath, cause);
    }
}

export class FileReadError extends FileSystemError {
    constructor(filePath: string, cause?: Error) {
        super('Error reading file', filePath, cause);
    }
}

export class FileImportError extends FileSystemError {
    constructor(filePath: string, cause?: Error) {
        super('Error importing file', filePath, cause);
    }
}

// Tool Errors
export class ToolError extends BaseError {
    constructor(message: string, public readonly toolName: string, cause?: Error) {
        super(`${message}: ${toolName}`, cause);
    }
}

export class InvalidToolError extends ToolError {
    constructor(toolName: string, details: string, cause?: Error) {
        super(`Invalid tool found: ${details}`, toolName, cause);
    }
}

export class ToolNotFoundError extends ToolError {
    constructor(toolName: string, cause?: Error) {
        super('Tool not found', toolName, cause);
    }
}

export class FunctionCallError extends ToolError {
    constructor(functionName: string, details: string, cause?: Error) {
        super(`Error calling function: ${details}`, functionName, cause);
    }
}

// API Errors
export class APIError extends BaseError {
    constructor(
        message: string,
        public readonly payload: ChatCompletionCreateParamsNonStreaming,
        cause?: Error
    ) {
        super(`${message}. Payload: ${JSON.stringify(payload)}`, cause);
    }
}

export class ToolCompletionError extends APIError {
    constructor(payload: ChatCompletionCreateParamsNonStreaming, cause?: Error) {
        super('Tool completion failed', payload, cause);
    }
}

export class ChatCompletionError extends APIError {
    constructor(payload: ChatCompletionCreateParamsNonStreaming, cause?: Error) {
        super('Chat completion failed', payload, cause);
    }
}

// Storage Errors
export class StorageError extends BaseError {
    constructor(message: string, cause?: Error) {
        super(message, cause);
    }
}

export class RedisError extends StorageError {
    constructor(message: string, cause?: Error) {
        super(`Redis error: ${message}`, cause);
    }
}

export class RedisConnectionError extends RedisError {
    constructor(message: string, cause?: Error) {
        super(message, cause);
    }
}

export class RedisKeyValidationError extends RedisError {
    constructor(message: string, cause?: Error) {
        super(`Key validation failed: ${message}`, cause);
    }
}
