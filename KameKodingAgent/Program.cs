using Microsoft.Extensions.AI;
using System.ClientModel;
using System.CommandLine;
using System.CommandLine.Parsing;
using System.ComponentModel;
using System.Text;

namespace KameKodingAgent;

enum LlmBackend
{
    VertexAi,
    Anthropic,
    Ollama,
    // The new responses API. More efficient for the official OpenAI api, but less compatible.
    OpenAiResponses,
    // The original chat API, widely supported by other API providers like llama.cpp.
    OpenAiChat,
}

internal class Program
{
    // TODO: make configurable
    const string GCP_PROJECT_ID = "ai-test-414105";

    static async Task<int> Main(string[] args)
    {
        Option<string> rootDirectoryOption = new("--root-directory")
        {
            Description = "What root directory to use, defaults to current directory.",
            DefaultValueFactory = _ => Environment.CurrentDirectory,
        };

        Option<LlmBackend> llmBackendOption = new("--llm-backend")
        {
            Description = "Which LLM backend to use.",
            DefaultValueFactory = _ => LlmBackend.VertexAi,
        };

        Option<string> modelOption = new("--model")
        {
            Description = "Which model to use. How this is interpreted is based on which LLM is used.",
            DefaultValueFactory = a => a.GetRequiredValue(llmBackendOption) switch
            {
                LlmBackend.VertexAi => $"projects/{GCP_PROJECT_ID}/locations/global/publishers/google/models/gemini-3.7-flash",
                LlmBackend.Anthropic => "claude-haiku-5-5",
                LlmBackend.Ollama => "gemma4:12b",
                LlmBackend.OpenAiResponses => "gpt-6-luna",
                LlmBackend.OpenAiChat => "gpt-6-luna",
                _ => throw new Exception("Programming error: unhandled LLM backend."),
            },
        };

        Option<string?> endpointOption = new("--endpoint")
        {
            Description = "Which endpoint to use with for the API.",
            DefaultValueFactory = a => a.GetRequiredValue(llmBackendOption) switch
            {
                LlmBackend.Ollama => "http://localhost:11434",
                _ => null,
            },
        };

        var rootCommand = new RootCommand("KameKodingAgent");
        rootCommand.Options.Add(rootDirectoryOption);
        rootCommand.Options.Add(llmBackendOption);
        rootCommand.Options.Add(modelOption);
        rootCommand.Options.Add(endpointOption);

        ParseResult parseResult;
        try
        {
            parseResult = rootCommand.Parse(args);
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.Message);
            return 1;
        }
        if (parseResult.Errors.Count != 0)
        {
            Console.WriteLine("Failed to parse args.");
            foreach (ParseError parseError in parseResult.Errors)
            {
                Console.Error.WriteLine(parseError.Message);
            }
            return 1;
        }

        string rootDirectory = parseResult.GetRequiredValue(rootDirectoryOption);
        string modelName = parseResult.GetRequiredValue(modelOption);
        string? endpoint = parseResult.GetRequiredValue(endpointOption);
        IChatClient chatClient = parseResult.GetRequiredValue(llmBackendOption) switch
        {
            LlmBackend.VertexAi => CreateVertexAiChatClient(),
            LlmBackend.Anthropic => new Anthropic.AnthropicClient().AsIChatClient(),
            LlmBackend.Ollama => new OllamaSharp.OllamaApiClient(endpoint!),
            LlmBackend.OpenAiResponses => CreateOpenAiResponseClient(endpoint),
            LlmBackend.OpenAiChat => CreateOpenAiChatClient(modelName, endpoint),
            _ => throw new Exception("Programming error: unhandled LLM backend."),
        };

        var program = new Program(chatClient, rootDirectory, parseResult.GetRequiredValue(llmBackendOption), modelName);
        await program.Run();
        return 0;
    }

#pragma warning disable OPENAI001 // Type is for evaluation purposes only and is subject to change or removal in future updates.
    private static IChatClient CreateOpenAiResponseClient(string? endpoint)
    {
        const string VAR_NAME = "OPENAI_API_KEY";
        string? apiKey = Environment.GetEnvironmentVariable(VAR_NAME);
        if (string.IsNullOrEmpty(apiKey))
        {
            throw new ArgumentException($"Environmental variable {VAR_NAME} not set.");
        }
        var cred = new ApiKeyCredential(apiKey);
        var options = new OpenAI.Responses.ResponsesClientOptions();
        if (endpoint != null)
        {
            options.Endpoint = new Uri(endpoint);
        }
        var client = new OpenAI.Responses.ResponsesClient(cred, options);
        return client.AsIChatClient();
    }
#pragma warning restore OPENAI001 // Type is for evaluation purposes only and is subject to change or removal in future updates.

    private static IChatClient CreateOpenAiChatClient(string modelName, string? endpoint)
    {
        const string VAR_NAME = "OPENAI_API_KEY";
        string? apiKey = Environment.GetEnvironmentVariable(VAR_NAME);
        if (string.IsNullOrEmpty(apiKey))
        {
            throw new ArgumentException($"Environmental variable {VAR_NAME} not set.");
        }
        var cred = new ApiKeyCredential(apiKey);
        var options = new OpenAI.OpenAIClientOptions();
        if (endpoint != null)
        {
            options.Endpoint = new Uri(endpoint);
        }
        var client = new OpenAI.Chat.ChatClient(modelName, cred, options);
        return client.AsIChatClient();
    }

    private static IChatClient CreateVertexAiChatClient()
    {
        var builder = new Google.Cloud.AIPlatform.V1.PredictionServiceClientBuilder()
        {
            QuotaProject = GCP_PROJECT_ID,
        };
        return Google.Cloud.VertexAI.Extensions.VertexAIExtensions.BuildIChatClient(builder);
    }

    private readonly string _rootPath;
    private readonly ChatOptions _options;
    private readonly IChatClient _chatClient;
    private readonly ConsoleColor _defaultForeColor;
    private readonly List<ChatMessage> _conversation;
    private readonly LlmBackend _backend;

    private Program(IChatClient chatClient, string rootPath, LlmBackend backend, string modelName)
    {
        rootPath = Path.GetFullPath(rootPath);
        if (rootPath.EndsWith(Path.DirectorySeparatorChar))
            rootPath = rootPath.Substring(0, rootPath.Length - 1);
        if (!Directory.Exists(rootPath))
        {
            throw new ArgumentException("Directory does not exist: " + rootPath);
        }
        _rootPath = rootPath;
        _options = new ChatOptions()
        {
            ModelId = modelName,
            Tools =
            [
                AIFunctionFactory.Create(ListFiles),
                AIFunctionFactory.Create(ReadFile),
                AIFunctionFactory.Create(WriteFile),
            ],
            ToolMode = ChatToolMode.Auto,
        };
        if (backend == LlmBackend.Anthropic)
        {
            _options.MaxOutputTokens = 4000;
        }
        _chatClient = chatClient.AsBuilder().UseFunctionInvocation().Build();
        _defaultForeColor = Console.ForegroundColor;
        _conversation = new List<ChatMessage>();
        _backend = backend;
    }

    void ResetConversation()
    {
        _conversation.Clear();
        _conversation.Add(new ChatMessage(ChatRole.System, "You are a programmer who edits code files based on instructions. Before editing a file, read its contents to figure out what needs to be replaced."));
    }

    async Task Run()
    {
        Console.WriteLine($"KameKodingAgent, using {_backend} with model {_options.ModelId}, running in: " + _rootPath);

        ResetConversation();
        while (true)
        {
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Please enter your prompt. Enter a blank line to finish the prompt.");
            Console.ForegroundColor = _defaultForeColor;

            StringBuilder sb = new StringBuilder();
            while (true)
            {
                string? line = Console.ReadLine()?.Trim();
                if (line == null || line == "/exit")
                {
                    Console.WriteLine("Exiting.");
                    return;
                }
                else if (line == "/clear")
                {
                    Console.WriteLine("Cleaning context.");
                    ResetConversation();
                    continue;
                }
                else if (line.Length == 0)
                {
                    break;
                }
                sb.AppendLine(line);
            }

            string prompt = sb.ToString().Trim();

            await ProcessUserPrompt(prompt);
        }
    }

    private async Task ProcessUserPrompt(string prompt)
    {
        _conversation.Add(new ChatMessage(ChatRole.User, prompt));

        var updates = new List<ChatResponseUpdate>();
        bool? isThinking = null;
        string? prevRole = null;
        await foreach (var update in _chatClient.GetStreamingResponseAsync(_conversation, _options))
        {
            if (update.ConversationId == null)
            {
                updates.Add(update);
            }
            else
            {
                _options.ConversationId = update.ConversationId;
            }

            if (update.Role.HasValue && (prevRole is null || prevRole != update.Role.Value.Value))
            {
                string role = update.Role.Value.Value;
                Console.WriteLine();
                Console.ForegroundColor = GetColorForRole(role);
                Console.Write($"{role}: ");
                Console.ForegroundColor = _defaultForeColor;
                prevRole = role;
            }
            foreach (var content in update.Contents)
            {
                if (content is TextContent textContent)
                {
                    if (isThinking == true)
                    {
                        Console.WriteLine();
                    }
                    isThinking = false;
                    Console.Write(textContent.Text);
                }
                else if (content is TextReasoningContent reasoningContent)
                {
                    if (isThinking == false)
                    {
                        Console.WriteLine();
                    }
                    isThinking = true;
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.Write(reasoningContent.Text);
                    Console.ForegroundColor = _defaultForeColor;
                }
                else if (content is FunctionCallContent functionCallContent)
                {
                    isThinking = null;
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"<function-call name='{functionCallContent.Name}' id='{functionCallContent.CallId}'>");
                    if (functionCallContent?.Arguments is object)
                    {
                        foreach (var kvp in functionCallContent.Arguments)
                        {
                            Console.WriteLine($"\t<{kvp.Key}>{kvp.Value}</{kvp.Key}");
                        }
                    }
                    Console.WriteLine("</function-call>");
                    Console.ForegroundColor = _defaultForeColor;
                }
                else if (content is FunctionResultContent functionResultContent)
                {
                    isThinking = null;
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"<function-result id='{functionResultContent.CallId}'>{functionResultContent.Result?.ToString()}</function-result>");
                    Console.ForegroundColor = _defaultForeColor;
                }
                else if (content is UsageContent)
                {
                    // Don't care
                }
                else
                {
                    isThinking = null;
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine(content.GetType().Name);
                    Console.ForegroundColor = _defaultForeColor;
                }
            }
        }

        if (_options.ConversationId == null)
        {
            // Stateless API
            _conversation.AddRange(updates.ToChatResponse().Messages);
        }
        else
        {
            // Stateful API
            _conversation.Clear();
        }
    }

    private ConsoleColor GetColorForRole(string? role)
    {
        if (role == ChatRole.Assistant.Value)
        {
            return ConsoleColor.Red;
        }
        else if (role == ChatRole.Tool.Value)
        {
            return ConsoleColor.Blue;
        }
        else
        {
            return _defaultForeColor;
        }
    }

    private string NormalizePath(string path)
    {
        if (!Path.IsPathFullyQualified(path))
        {
            path = Path.Combine(_rootPath, path);
        }
        path = Path.GetFullPath(path);

        string? remainingPath = path;
        while (remainingPath != null)
        {
            if (remainingPath == _rootPath)
            {
                // Found that we are in the root directory, so it should be ok to read or write this file.
                return path;
            }

            string? fileName = Path.GetFileName(path);
            if (fileName != null && fileName[0] == '.')
            {
                // Found that we are in the root directory, so it should be ok to read or write this file.
                throw new Exception("File path contains a segment whose name starts with a dot: " + path);
            }

            remainingPath = Path.GetDirectoryName(remainingPath);
        }

        throw new Exception($"Path '{path}' not contained in the root directory '{_rootPath}'.");
    }

    [Description("List the files and directories in a directory.")]
    string ListFiles(string path)
    {
        path = NormalizePath(path);
        var di = new DirectoryInfo(path);

        StringBuilder sb = new StringBuilder();

        foreach (var fi in di.GetFileSystemInfos())
        {
            sb.Append(fi.Name);
            if (fi is DirectoryInfo)
            {
                sb.Append(Path.DirectorySeparatorChar);
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }

    [Description("Reads the contents of a file.")]
    string ReadFile(string path)
    {
        return File.ReadAllText(NormalizePath(path));
    }

    [Description("Writes content to a file.")]
    string WriteFile(string path, string newContents)
    {
        path = NormalizePath(path);
        File.WriteAllText(path, newContents);
        return "OK";
    }
}
