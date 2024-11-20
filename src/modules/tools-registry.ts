import path from 'path';
import * as fs from 'fs/promises';
import { OpenAIAgent } from '../agent';
import {
    AgentTools,
    ChatCompletionTool,
    FunctionDefinition,
    ToolChoices,
    ToolFunction,
    ToolFunctions,
} from '../types';

import {
    ValidationError,
    ToolNotFoundError,
    DirectoryAccessError,
    FileReadError,
    FileImportError,
    InvalidToolError,
    ToolConfigurationError,
} from '../errors';

/**
 * Singleton class for managing the tools registry.
 */
export class ToolsRegistry {
    private static instance: AgentTools | null = null;

    /**
     * Gets the current instance of the tools registry.
     * @returns {AgentTools | null} The current tools instance or null if not set.
     */
    static getInstance(): AgentTools | null {
        return ToolsRegistry.instance;
    }

    /**
     * Sets the instance of the tools registry.
     * @param {AgentTools} tools - The tools instance to set.
     */
    static setInstance(tools: AgentTools): void {
        ToolsRegistry.instance = tools;
    }
}

/**
 * Validates the function name.
 * @param {string} name - The name of the function to validate.
 * @throws {ValidationError} If the function name is invalid.
 */
const validateFunctionName = (name: string): void => {
    if (!name || typeof name !== 'string') {
        throw new ValidationError('Function name must be a non-empty string');
    }
    if (name.length > 64) {
        throw new ValidationError('Function name must not exceed 64 characters');
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
        throw new ValidationError('Function name must contain only alphanumeric characters, underscores, and hyphens');
    }
};

/**
 * Validates the function parameters.
 * @param {unknown} params - The parameters of the function to validate.
 * @throws {ValidationError} If the parameters are invalid.
 */
const validateFunctionParameters = (params: unknown): void => {
    if (!params || typeof params !== 'object') {
        throw new ValidationError('Function parameters must be a non-null object');
    }
};

/**
 * Validates the function definition.
 * @param {unknown} func - The function definition to validate.
 * @throws {ValidationError} If the function definition is invalid.
 */
const validateFunctionDefinition = (func: unknown): void => {
    if (!func || typeof func !== 'object') {
        throw new ValidationError('Function definition must be a non-null object');
    }

    const { name, description, parameters, strict } = func as FunctionDefinition;

    validateFunctionName(name);

    if (description !== undefined && typeof description !== 'string') {
        throw new ValidationError('Function description must be a string when provided');
    }

    if (parameters !== undefined) {
        validateFunctionParameters(parameters);
    }

    if (strict !== undefined && strict !== null && typeof strict !== 'boolean') {
        throw new ValidationError('Function strict flag must be a boolean when provided');
    }
};

/**
 * Validates a chat completion tool definition.
 * @param {unknown} tool - The tool definition to validate.
 * @throws {ValidationError} If the tool definition is invalid.
 */
const validateChatCompletionTool = (tool: unknown): void => {
    if (!tool || typeof tool !== 'object') {
        throw new ValidationError('Chat completion tool must be a non-null object');
    }

    const { function: functionDefinition, type } = tool as ChatCompletionTool;

    if (type !== 'function') {
        throw new ValidationError('Chat completion tool type must be "function"');
    }

    validateFunctionDefinition(functionDefinition);
};

/**
 * Validates the configuration of tools and their definitions.
 * @param {ChatCompletionTool[]} fnDefinitions - The array of tool definitions.
 * @param {ToolFunctions} functions - The object containing tool implementations.
 * @throws {ValidationError | ToolNotFoundError} If there is a mismatch in definitions or missing implementations.
 */
const validateToolConfiguration = (
    fnDefinitions: ChatCompletionTool[],
    functions: ToolFunctions
): void => {
    const definitionCount = fnDefinitions.length;
    const functionCount = Object.keys(functions).length;

    if (definitionCount !== functionCount) {
        throw new ValidationError(
            `Mismatch between number of function definitions (${definitionCount}) and implementations (${functionCount})`
        );
    }

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
 * Loads tool files from a specified directory.
 * @param {string} dirPath - The path to the directory containing tool files.
 * @returns {Promise<AgentTools>} A promise that resolves to the loaded agent tools.
 * @throws {DirectoryAccessError | FileReadError | FileImportError | InvalidToolError | ToolConfigurationError | ValidationError} If an error occurs during loading.
 */
export const loadToolsDirFiles = async (
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
                    if (error instanceof ValidationError) {
                        throw new InvalidToolError(
                            fullPath,
                            fnName,
                            `Invalid tool definition: ${error.message}`
                        );
                    }
                    throw new InvalidToolError(
                        fullPath,
                        fnName,
                        'Unexpected error validating tool'
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
            error instanceof ToolConfigurationError ||
            error instanceof ValidationError
        ) {
            throw error;
        }

        throw new Error(
            `Unexpected error loading tools: ${
                error instanceof Error ? error.message : 'Unknown error'
            }`
        );
    }
};

/**
 * Imports tool functions based on their names.
 * @param {string[]} toolNames - An array of tool names to import.
 * @returns {Promise<ToolChoices>} A promise that resolves to the imported tool functions and choices.
 * @throws {ValidationError | ToolNotFoundError} If the tools directory path is not set or tools are missing.
 */
export const importToolFunctions = async (
    toolNames: string[]
): Promise<ToolChoices> => {
    try {
        if (!OpenAIAgent.toolsDirPath) {
            throw new ValidationError(
                'Tools directory path not set. Call loadToolsDirFiles with your tools directory path first.'
            );
        }

        const tools =
            ToolsRegistry.getInstance() ??
            (await loadToolsDirFiles(OpenAIAgent.toolsDirPath));

        const toolChoices = toolNames
            .map((toolName) =>
                tools.toolDefinitions.find(
                    (tool) => tool.function.name === toolName
                )
            )
            .filter((tool): tool is ChatCompletionTool => tool !== undefined);

        const missingTools = toolNames.filter(
            (name) =>
                !toolChoices.some((tool) => tool.function.name === name)
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
            error instanceof ValidationError ||
            error instanceof ToolNotFoundError
        ) {
            throw error;
        }

        throw new Error(
            `Failed to import tool functions: ${
                error instanceof Error ? error.message : 'Unknown error'
            }`
        );
    }
};
