import path from 'path';
import * as fs from 'fs/promises';
import { OpenAIAgent } from '../agent';

import { ChatCompletionTool } from 'openai/src/resources/index.js';
import {
    FunctionDefinition,
    FunctionParameters,
} from 'openai/src/resources/shared.js';


type ToolFunction = (args: object) => Promise<string> | string | undefined;

export interface ToolFunctions {
    [key: string]: ToolFunction;
}

export interface AgentTools {
    toolSchemas: ChatCompletionTool[];
    toolFunctions: ToolFunctions;
}

export interface ToolChoices {
    toolChoices: ChatCompletionTool[];
    toolFunctions: ToolFunctions;
}

export class ToolsRegistry {
    private static instance: AgentTools | null = null;

    static getInstance(): AgentTools | null {
        return ToolsRegistry.instance;
    }

    static setInstance(tools: AgentTools): void {
        ToolsRegistry.instance = tools;
    }
}


const isValidFunctionName = (name: string): boolean => {
    return /^[a-zA-Z0-9_-]+$/.test(name) && name.length <= 64;
};

const isValidFunctionParameters = (
    params: unknown
): params is FunctionParameters => {
    return typeof params === 'object' && params !== null;
};

const isValidFunctionDefinition = (
    func: unknown
): func is FunctionDefinition => {
    if (typeof func !== 'object' || func === null) {
        return false;
    }

    const { name, description, parameters, strict } =
        func as FunctionDefinition;

    if (typeof name !== 'string' || !isValidFunctionName(name)) {
        return false;
    }

    if (description !== undefined && typeof description !== 'string') {
        return false;
    }

    if (parameters !== undefined && !isValidFunctionParameters(parameters)) {
        return false;
    }

    if (
        strict !== undefined &&
        strict !== null &&
        typeof strict !== 'boolean'
    ) {
        return false;
    }

    return true;
};

const isValidChatCompletionTool = (
    tool: unknown
): tool is ChatCompletionTool => {
    if (typeof tool !== 'object' || tool === null) {
        return false;
    }

    const { function: functionDefinition, type } = tool as ChatCompletionTool;

    if (type !== 'function') {
        return false;
    }

    return isValidFunctionDefinition(functionDefinition);
};

const validateToolConfiguration = (
    schemas: ChatCompletionTool[],
    functions: ToolFunctions
): void => {
    if (Object.keys(schemas).length !== Object.keys(functions).length) {
        throw new Error('Mismatch between number of schemas and functions');
    }

    for (const schema of schemas) {
        if (!functions[schema.function.name]) {
            throw new Error(
                `Missing function implementation for schema: ${schema.function.name}`
            );
        }
    }
};

export const loadToolFunctions = async (
    dirPath: string
): Promise<AgentTools> => {
    const toolSchemas: ChatCompletionTool[] = [];
    const toolFunctions: ToolFunctions = {};

    try {
        await fs.access(dirPath);

        const files = await fs.readdir(dirPath);

        for (const file of files) {
            if (!file.endsWith('.js') && !file.endsWith('.ts')) continue;

            const fullPath = path.join(dirPath, file);
            const stat = await fs.stat(fullPath);

            if (stat.isFile()) {
                const fileFunctions = await import(fullPath);

                for (const [fnName, fn] of Object.entries(fileFunctions)) {
                    if (typeof fn === 'function') {
                        toolFunctions[fnName] = fn as ToolFunction;
                    } else if (isValidChatCompletionTool(fn)) {
                        toolSchemas.push(fn as ChatCompletionTool);
                    } else {
                        console.error(
                            `Invalid tool schema or function found: ${fnName}`
                        );
                    }
                }
            }
        }

        validateToolConfiguration(toolSchemas, toolFunctions);

        const tools = { toolSchemas, toolFunctions };
        ToolsRegistry.setInstance(tools);
        return tools;
    } catch (error) {
        const errorMessage =
            error instanceof Error ? error.message : 'Unknown error';
        throw new Error(
            `Failed to load tool functions from ${dirPath}: ${errorMessage}`
        );
    }
};

export const importToolFunctions = async (
    toolNames: string[]
): Promise<ToolChoices> => {
    try {
        if (!OpenAIAgent.toolsDirPath)
            throw new Error('Tools have not been loaded yet');
        const tools =
            ToolsRegistry.getInstance() ??
            (await loadToolFunctions(OpenAIAgent.toolsDirPath));

        const toolChoices = toolNames
            .map((toolName) =>
                tools.toolSchemas.find(
                    (tool) => tool.function.name === toolName
                )
            )
            .filter((tool): tool is ChatCompletionTool => tool !== undefined);

        if (toolChoices.length !== toolNames.length) {
            const missingTools = toolNames.filter(
                (name) =>
                    !toolChoices.some((tool) => tool.function.name === name)
            );
            throw new Error(`Tools not found: ${missingTools.join(', ')}`);
        }

        return {
            toolFunctions: tools.toolFunctions,
            toolChoices,
        };
    } catch (error) {
        throw new Error(
            `Failed to get tool functions: ${
                error instanceof Error ? error.message : 'Unknown error'
            }`
        );
    }
};
