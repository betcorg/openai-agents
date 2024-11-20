export class ValidationError extends Error {
    constructor(message?: string) {
        super(message);
        this.name = 'ValidationError';
    }
}

export class ToolNotFoundError extends Error {
    constructor(toolName: string) {
        super(`Tool not found: ${toolName}`);
        this.name = 'ToolNotFoundError';
    }
}

export class DirectoryAccessError extends Error {
    constructor(dirPath: string, cause?: Error) {
        super(`Was not possible to access the directory: ${dirPath}`);
        this.name = 'DirectoryAccessError';
        this.cause = cause;
    }
}

export class FileReadError extends Error {
    constructor(filePath: string, cause?: Error) {
        super(`Error reading file: ${filePath}`);
        this.name = 'FileReadError';
        this.cause = cause;
    }
}

export class FileImportError extends Error {
    constructor(filePath: string, cause?: Error) {
        super(`Error importing file: ${filePath}`);
        this.name = 'FileImportError';
        this.cause = cause;
    }
}

export class InvalidToolError extends Error {
    constructor(filePath: string, functionName: string, message: string ) {
        super(`Invalid tool found at ${filePath}: ${functionName}. ${message}`);
        this.name = 'InvalidToolError';
    }
}

export class ToolConfigurationError extends Error {
    constructor(message: string) {
        super(`Error on the tools configuration: ${message}`);
        this.name = 'ToolConfigurationError';
    }
}