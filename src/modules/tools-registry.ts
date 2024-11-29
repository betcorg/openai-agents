import path from 'path';
import * as fs from 'fs/promises';
import { AgentTools, ToolChoices, ToolFunction, ToolFunctions } from '../types';

import {
    ValidationError,
    ToolNotFoundError,
    DirectoryAccessError,
    FileReadError,
    FileImportError,
    InvalidToolError,
} from '../errors';

import { ChatCompletionTool, FunctionDefinition } from 'openai/resources';

/**
 * @class ToolsRegistry
 * @description Singleton class for managing the tools registry.  Holds the currently loaded tools.
 */
export class ToolsRegistry {
    private static instance: AgentTools | null = null;
    public static toolsDirPath: string | null = null;

    /**
     * Gets the current instance of the tools registry.
     */
    static getInstance(): AgentTools | null {
        return ToolsRegistry.instance;
    }

    /**
     * Sets the instance of the tools registry.
     */
    static setInstance(tools: AgentTools): void {
        ToolsRegistry.instance = tools;
    }
}

/**
 * Validates the function name, ensuring it meets OpenAI's requirements.
 */
const validateFunctionName = (name: string): void => {
    if (!name || typeof name !== 'string') {
        throw new InvalidToolError(
            name,
            'Function name must be a non-empty string'
        );
    }
    if (name.length > 64) {
        throw new InvalidToolError(
            name,
            'Function name must not exceed 64 characters'
        );
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
        throw new InvalidToolError(
            name,
            'Function name must contain only alphanumeric characters, underscores, and hyphens'
        );
    }
};

/**
 * Validates the function parameters, ensuring they are a non-null object.
 */
const validateFunctionParameters = (params: unknown, name: string): void => {
    if (!params || typeof params !== 'object') {
        throw new InvalidToolError(
            name,
            'Function parameters must be a non-null object'
        );
    }
};

/**
 * Validates the function definition,
 * checking name, description, parameters, and strict flag.
 */
const validateFunctionDefinition = (func: unknown): void => {
    const { name, description, parameters, strict } =
        func as FunctionDefinition;

    if (!func || typeof func !== 'object') {
        throw new InvalidToolError(
            name,
            'Function definition must be a non-null object'
        );
    }

    validateFunctionName(name);

    if (description !== undefined && typeof description !== 'string') {
        throw new InvalidToolError(
            name,
            'Function description must be a string when provided'
        );
    }

    if (parameters !== undefined) {
        validateFunctionParameters(parameters, name);
    }

    if (
        strict !== undefined &&
        strict !== null &&
        typeof strict !== 'boolean'
    ) {
        throw new InvalidToolError(
            name,
            'Function strict flag must be a boolean when provided'
        );
    }
};

/**
 * Validates a chat completion tool definition,
 * ensuring it has the correct type and a valid function definition.
 */
const validateChatCompletionTool = (tool: unknown): void => {
    const {
        function: functionDefinition,
        function: { name },
        type,
    } = tool as ChatCompletionTool;

    if (!tool || typeof tool !== 'object') {
        throw new InvalidToolError(
            name,
            'Chat completion tool must be a non-null object'
        );
    }

    if (type !== 'function') {
        throw new InvalidToolError(
            name,
            'Chat completion tool type must be "function"'
        );
    }

    validateFunctionDefinition(functionDefinition);
};

/**
 * Validates the configuration of tools, ensuring that all defined tools have corresponding implementations.
 */
const validateToolConfiguration = (
    fnDefinitions: ChatCompletionTool[],
    functions: ToolFunctions
): void => {
    for (const def of fnDefinitions) {
        const functionName = def.function.name;
        if (!functions[functionName]) {
            throw new ToolNotFoundError(
                `Missing function implementation for tool: ${functionName}`
            );
        }
    }
};

/**
 * Loads tool files (both definitions and implementations) from a specified directory.
 *
 * @param {string} dirPath - The path to the directory containing tool files.
 * @returns {Promise<AgentTools>} A promise that resolves to the loaded agent tools.
 * @throws {DirectoryAccessError | FileReadError | FileImportError | InvalidToolError | ToolNotFoundError} If an error occurs during loading.
 */
export const loadToolsDirFunctions = async (
    dirPath: string
): Promise<AgentTools> => {
    try {
        // Validate directory access
        try {
            await fs.access(dirPath);
        } catch (error) {
            throw new DirectoryAccessError(
                dirPath,
                error instanceof Error ? error : undefined
            );
        }

        // Read directory contents
        let files: string[];
        try {
            files = await fs.readdir(dirPath);
        } catch (error) {
            throw new FileReadError(
                dirPath,
                error instanceof Error ? error : undefined
            );
        }

        const toolDefinitions: ChatCompletionTool[] = [];
        const toolFunctions: ToolFunctions = {};

        // Process each file
        for (const file of files) {
            if (!file.endsWith('.js') && !file.endsWith('.ts')) continue;

            const fullPath = path.join(dirPath, file);

            // Validate file status
            try {
                const stat = await fs.stat(fullPath);
                if (!stat.isFile()) continue;
            } catch (error) {
                throw new FileReadError(
                    fullPath,
                    error instanceof Error ? error : undefined
                );
            }

            // Import file contents
            let fileFunctions;
            try {
                fileFunctions = await import(fullPath);
            } catch (error) {
                throw new FileImportError(
                    fullPath,
                    error instanceof Error ? error : undefined
                );
            }

            // Process functions
            const funcs = fileFunctions.default || fileFunctions;
            for (const [fnName, fn] of Object.entries(funcs)) {
                try {
                    if (typeof fn === 'function') {
                        toolFunctions[fnName] = fn as ToolFunction;
                    } else {
                        // Validate as tool definition
                        validateChatCompletionTool(fn);
                        toolDefinitions.push(fn as ChatCompletionTool);
                    }
                } catch (error) {
                    if (error instanceof InvalidToolError) throw error;

                    throw new InvalidToolError(
                        fnName,
                        `Unexpected error validating tool: ${
                            error instanceof Error
                                ? error.message
                                : 'Unknown error'
                        }`
                    );
                }
            }
        }

        // Validate final configuration
        validateToolConfiguration(toolDefinitions, toolFunctions);

        const tools = { toolDefinitions, toolFunctions };
        ToolsRegistry.setInstance(tools);
        return tools;
    } catch (error) {
        if (
            error instanceof DirectoryAccessError ||
            error instanceof FileReadError ||
            error instanceof FileImportError ||
            error instanceof InvalidToolError ||
            error instanceof ToolNotFoundError
        )
            throw error;

        throw new Error(
            `Unexpected error loading tools: ${
                error instanceof Error ? error.message : 'Unknown error'
            }`
        );
    }
};

/**
 * Imports and returns specific tool functions based on their names.
 * Loads tools from the directory if they haven't been loaded yet.
 *
 * @param {string[]} toolNames - An array of tool names to import.
 * @returns {Promise<ToolChoices>} A promise that resolves to the imported tool functions and choices.
 * @throws {ValidationError | ToolNotFoundError | InvalidToolError} If the tools directory path is not set or if any requested tools are missing.
 */
export const importToolFunctions = async (
    toolNames: string[]
): Promise<ToolChoices> => {
    try {
        if (!ToolsRegistry.toolsDirPath) {
            throw new ValidationError(
                'Tools directory path not set. Call loadToolsDirFunctions with your tools directory path first.'
            );
        }

        const tools =
            ToolsRegistry.getInstance() ??
            (await loadToolsDirFunctions(ToolsRegistry.toolsDirPath));

        const toolChoices = toolNames
            .map((toolName) =>
                tools.toolDefinitions.find(
                    (tool) => tool.function.name === toolName
                )
            )
            .filter((tool): tool is ChatCompletionTool => tool !== undefined);

        const missingTools = toolNames.filter(
            (name) => !toolChoices.some((tool) => tool.function.name === name)
        );

        if (missingTools.length > 0) {
            throw new ToolNotFoundError(
                `The following tools were not found: ${missingTools.join(', ')}`
            );
        }

        return {
            toolFunctions: tools.toolFunctions,
            toolChoices,
        };
    } catch (error) {
        if (
            error instanceof DirectoryAccessError ||
            error instanceof FileReadError ||
            error instanceof FileImportError ||
            error instanceof InvalidToolError ||
            error instanceof ValidationError ||
            error instanceof ToolNotFoundError
        )
            throw error;

        throw new Error(
            `Failed to import tool functions: ${
                error instanceof Error ? error.message : 'Unknown error'
            }`
        );
    }
};
