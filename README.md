# OpenAI-Agents

A TypeScript library that extends the functionality of the official OpenAI SDK, enabling enhanced chat completions and tool integrations.

## Features

-   **Customizable AI Agents**: Easily create AI chatbots with configurable parameters.
-   **Tool Integration**: Seamlessly load and utilize external tool functions.
-   **Chat History Management**: Store and retrieve chat history using Redis.
-   **Completion Usage Management**: Get the total token count of the completions made.

## Installation

```bash
npm install openai-agents
```

## Usage

### Initialization

To create an instance of `OpenAIAgent`, provide the required options:

```javascript
import { OpenAIAgent } from 'openai-agents';

// Set default configurations for the agent instance:
const agent = new OpenAIAgent({
    model: 'gpt-4o-mini', // The model is required to initialize the agent.
    temperature: 0.7,
    max_tokens: 4000,
    system_instruction: 'Your system instruction here',
});

/*
You can pass default messages to be appended
at the beginning of the chat history for "few-shot" prompting.
*/
const agent = new OpenAIAgent({
    messages: [
        { role: 'user', content: 'Example of your query' },
        { role: 'assistant', content: 'Example of the expected answer' },
    ],
});

/* 
This library provides convenient setters and getters
for configuring and retrieving agent settings: 
*/
const temp = agent.temperature;
console.log(temp); // 0.7

agent.max_tokens = 300;
console.log(agent.max_tokens); // 300
```

### Creating a Chat Completion

Use the `createChatCompletion` method to generate chat responses:

```javascript
const result = await agent.createChatCompletion('Hi');

console.log(result);
/* 
Output:

{
  choices: ['Hi, how can I help you?'],
  total_usage: { prompt_tokens: 16, completion_tokens: 26, total_tokens: 42 },
  completion_messages: [
    { role: 'user', content: 'Hi' },
    {
      role: 'assistant',
      content: 'Hi, how can I help you?',
      refusal: null
    }
  ],
  completions: [
    ...completions
  ]
} 
*/

// You can customize the completion parameters for each request.
const resultWithCustomParams = await agent.createChatCompletion(
    'Explain AI in 20 words',
    // These parameters will override the default configurations only for this request.
    {
        system_instruction: 'Provide helpful responses.',
        custom_params: {
            max_tokens: 500,
            temperature: 1,
        },
    }
);

// If you want to produce more than one output
const resultWithMultipleOutputs = await agent.createChatCompletion(
    'Explain AI in 20 words',
    {
        system_instruction: 'Provide helpful responses.',
        custom_params: {
            n: 3, // The number of required outputs.
        },
    }
);

for (const choice of resultWithMultipleOutputs.choices) {
    console.log(choice);
}
/*
Output:

'AI, or artificial intelligence, is technology that enables machines to perform tasks typically requiring human intelligence, like learning and problem-solving.',
'AI, or artificial intelligence, mimics human cognitive functions through machines to perform tasks like learning, problem-solving, and decision-making.',
'AI, or artificial intelligence, simulates human-like intelligence in machines, enabling them to learn, reason, and solve complex problems autonomously.'

*/
```

Currently, only non-streaming completions are supported. This might change in the future.

### Function Calling

The agent can easily interact with external custom tools. To do this, you need to create a specific directory to store the files where your tools are defined. For more details, please refer to the official function calling [documentation](https://platform.openai.com/docs/guides/function-calling). Each JS/TS file can contain one or more function definitions along with their corresponding implementations. A file within your tools directory might look like this:

```javascript
// googleSearch.js

// Function definition
export const google_search = {
    type: 'function',
    function: {
        name: 'googleSearch', // Should match the name of your associated function.
        description: 'Search on Google for the given query',
        parameters: {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: 'The user search query',
                },
                file_type: {
                    type: 'string',
                    description:
                        'The format or extension of the document to look for',
                },
                search_results: {
                    type: 'number',
                    description:
                        'The number of results to return. The maximum is 10.',
                },
            },
            required: ['query'],
        },
    },
};

// Associated function
export const googleSearch = async (args) => {
    try {
        const { query, file_type, search_results } = args;
        // Your logic
        return results; // This should be a string
    } catch (error) {
        console.error('Error while running Google search.');
        throw error;
    }
};
```

### Loading Tool Functions

You can load your tools by running the `loadToolFunctions` method and passing your tools directory path:

```javascript
await agent.loadToolFunctions('./path/to/tools');
```

This method returns `true` if the tools were loaded correctly.

### Making Tool Requests

You can now use your tools by specifying their names in the `tool_choices` property. The agent will choose the appropriate tools for the given request from the loaded files.

```javascript
const result = await agent.createChatCompletion(
    'Search on Google "AI courses" and give me 5 results',
    {
        tool_choices: ['googleSearch'],
    }
);

console.log(result);
```

### Setting Up Redis Storage

To enable chat history storage, initialize the agent with a Redis client:

```bash
npm install redis
```

```javascript
import { createClient } from 'redis';

const redisClient = await createClient({
    url: 'redis://127.0.0.1:6379',
})
    .on('error', (err) => console.log('Redis Client Error', err))
    .connect();

/*
 You can optionally set default configurations
 for the agent's history management.
*/
await agent.setStorage(redisClient, {
    history: {
        /*
         The number of messages that will be sent to the model in the context messages array.
        */
        appended_messages: 6,

        /*
         Whether or not to send tool calls and function responses to the model in the context messages array.
         This property does not modify the database, only the retrieved messages.
        */
        send_tool_messages: true,

        /*
         Whether or not to remove tool messages (including tool calls and function responses) from the history stored in the database.
        */
        remove_tool_messages: true,

        /*
         The maximum number of messages stored for a specified user or session.
        */
        max_length: 100,

        /*
         The time (in seconds) that a chat history can live before being deleted.
        */
        ttl: 60 * 60 * 24,
    },
});
```

Now you can pass a `user` property in the `custom_params` object of your request, and the agent will store the conversation for that specific user. If you omit this `user` property, the agent will store the chat under a `default` key in a Redis list.

```javascript
const result = await agent.createChatCompletion(
    'Search on Google "AI courses" and give me 5 results',
    {
        custom_params: {
            user: '1234',
        },

    /*
     If the agent storage is enabled you can override your initial 
     history configurations for each request.
    */
        history: {
            appended_messages: 6,
            remove_tool_messages: true,
        },
    }
);
```

#### Deleting a user's history:

```javascript
await agent.deleteChatHistory(userID);
```

#### Retrieving a user's history:

```javascript
await agent.getChatHistory(userID);
```

## API Reference

### `OpenAIAgent`

#### Constructor

```javascript
constructor(agentOptions: AgentOptions, options?: ClientOptions)
```

-   **`agentOptions`**: Options for configuring the agent's behavior. These options correspond to the parameters of the OpenAI Chat Completion API (see the [API Reference](https://platform.openai.com/docs/api-reference/chat/create)). Note that `stream` and `stream_options` are disabled as only non-streaming completions are supported. Similarly, the deprecated `functions` and `function_call` options are not supported; use the `tools` parameter instead. You can also provide a system-level instruction via the `system_instruction` property and pre-populate messages using the `messages` property.
-   **`options`**: Client options for the underlying OpenAI API. Refer to the [openai-node](https://github.com/openai/openai-node/blob/master/src/index.ts) documentation for details.

#### Methods

##### Completions

-   **`createChatCompletion(options: { message: string; system_instruction?: string; tool_choices?: string[]; custom_params?: Partial<AgentCompletionParams>; history: HistoryOptions }): Promise<CompletionResult>`**: Generates a chat completion based on the provided user message and options.
    -   **`message: string`**: The user's input message.
    -   **`system_instruction?: string`**: An optional system-level instruction to guide the agent's response. This overrides any system instruction set in the constructor.
    -   **`tool_choices?: string[]`**: An optional array of tool names to restrict the agent's tool selection.
    -   **`custom_params?: Partial<AgentCompletionParams>`**: Allows customization of parameters passed to the underlying OpenAI Chat Completion API. These parameters correspond to the `ChatCompletionCreateParamsNonStreaming` interface, excluding `messages`, `stream`, `stream_options`, `function_call`, and `functions`. This allows you to adjust settings like `model`, `temperature`, `top_p`, etc.
    -   **`history: HistoryOptions`**: Options for managing conversation history.

##### Function Calling

-   **`loadToolFunctions(toolsDirAddr: string): Promise<boolean>`**: Loads tool functions from the specified directory.

    -   **`toolsDirAddr: string`**: The path to the directory containing the tool function definitions.

##### Chat History Management

-   **`setStorage(client: RedisClientType, options: { history: HistoryOptions }): Promise<boolean>`**: Configures Redis as the storage for chat history.

    -   **`client: RedisClientType`**: A Redis client instance.
    -   **`options: { history: HistoryOptions }`**: Options for managing the chat history stored in Redis.

-   **`deleteChatHistory(userId: string): Promise<boolean>`**: Deletes the chat history for the specified user from Redis.

    -   **`userId: string`**: The ID of the user whose history should be deleted.

-   **`getChatHistory(userId: string, options: HistoryOptions): Promise<ChatCompletionMessageParam[]>`**: Retrieves the chat history for the specified user from Redis.
    -   **`userId: string`**: The ID of the user whose history should be retrieved.
    -   **`options: HistoryOptions`**: Options for managing the retrieval of chat history.

Since the `OpenAIAgent` class extends the `OpenAI` class from the official SDK, all methods available in the OpenAI client are also available through the agent instance. For example, you can call the regular completion method:

```javascript
const response = await agent.chat.completions.create(params);
```

## Environment Variables

Ensure the following environment variable is set:

-   `OPENAI_API_KEY`: Your OpenAI API key.

If you pass an API key during agent initialization, it will override the environment variable, allowing you to use multiple agents with different keys for better token total_usage monitoring.

```javascript
const customerServiceAgent = new OpenAIAgent(
    {
        model: 'gpt-4o',
        max_tokens: 800,
        temperature: 0.5,
        system_instruction: 'You are a very helpful customer service agent.',
    },
    {
        apiKey: 'sk-my-customer-service-agent-key',
    }
);

const summarizer = new OpenAIAgent(
    {
        model: 'gpt-4o-mini',
        max_tokens: 4000,
        temperature: 0.5,
        system_instruction:
            'Your task is to summarize the given texts and extract the most important information.',
    },
    {
        apiKey: 'sk-my-summarizer-agent-key',
    }
);
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/betcorg/openai-agents/blob/master/LICENSE) file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## Contact

For questions or support, please contact hfranc.dev@gmail.com.
