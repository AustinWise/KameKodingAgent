using Microsoft.Extensions.AI;
using System.CommandLine;
using System.CommandLine.Parsing;
using System.ComponentModel;
using System.Text;

namespace KameKodingAgent;

internal class Program
{
    // TODO: make configurable
    const string GCP_PROJECT_ID = "ai-test-414105";
    const string GCP_REGION = "us-central1";

    enum LlmBackend
    {
        VertexAi,
        Anthropic,
        Ollama,
    }

    static async Task Main(string[] args)
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

        Option<string> modelNameOption = new("--model-name")
        {
            Description = "Which model to use. How this is interpreted is based on which LLM is used.",
            DefaultValueFactory = a => a.GetRequiredValue(llmBackendOption) switch
            {
                LlmBackend.VertexAi => $"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/publishers/google/models/gemini-2.5-pro",
                LlmBackend.Anthropic => "claude-opus-4-1-20250805",
                LlmBackend.Ollama => "qwen2.5-coder:7b",
            },
        };

        var rootCommand = new RootCommand("KameKodingAgent");
        rootCommand.Options.Add(rootDirectoryOption);
        rootCommand.Options.Add(llmBackendOption);
        rootCommand.Options.Add(modelNameOption);

        ParseResult parseResult = rootCommand.Parse(args);
        if (parseResult.Errors.Count != 0)
        {
            Console.WriteLine("Failed to parse args.");
            foreach (ParseError parseError in parseResult.Errors)
            {
                Console.Error.WriteLine(parseError.Message);
            }
            return;
        }

        string rootDirectory = parseResult.GetRequiredValue(rootDirectoryOption);
        string modelName = parseResult.GetRequiredValue(modelNameOption);
        IChatClient chatClient = parseResult.GetRequiredValue(llmBackendOption) switch
        {
            LlmBackend.VertexAi => CreateVertexAiChatClient(),
            LlmBackend.Anthropic => new Anthropic.SDK.AnthropicClient(Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY")).Messages,
            LlmBackend.Ollama => new OllamaSharp.OllamaApiClient("http://localhost:11434"),
        };

        var program = new Program(chatClient, rootDirectory, parseResult.GetRequiredValue(llmBackendOption), modelName);
        await program.Run();
    }

    private static IChatClient CreateVertexAiChatClient()
    {
        var builder = new Google.Cloud.AIPlatform.V1.PredictionServiceClientBuilder()
        {
            Endpoint = $"https://{GCP_REGION}-aiplatform.googleapis.com",
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

            _conversation.Add(new ChatMessage(ChatRole.User, prompt));

            var updates = new List<ChatResponseUpdate>();
            ChatRole? prevRole = null;
            await foreach (var update in _chatClient.GetStreamingResponseAsync(_conversation, _options))
            {
                if (update.Contents.Count == 0)
                    continue;

                updates.Add(update);
                if (!prevRole.HasValue || (prevRole.HasValue && update.Role.HasValue && prevRole.Value != update.Role.Value))
                {
                    if (prevRole.HasValue && updates.Count != 0)
                    {
                        _conversation.Add(new ChatMessage(prevRole.Value, [.. updates.SelectMany(u => u.Contents)]));
                        updates.Clear();
                    }

                    var role = update.Role!.Value;
                    Console.WriteLine();
                    Console.ForegroundColor = GetColorForRole(role);
                    Console.Write($"{role.Value}: ");
                    Console.ForegroundColor = _defaultForeColor;
                    prevRole = role;
                }
                foreach (var content in update.Contents)
                {
                    if (content is TextContent textContent)
                    {
                        Console.Write(textContent.Text);
                    }
                    else if (content is FunctionCallContent functionCallContent)
                    {
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
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine(content.GetType().Name);
                        Console.ForegroundColor = _defaultForeColor;
                    }
                }
            }

            if (prevRole.HasValue && updates.Count != 0)
            {
                _conversation.Add(new ChatMessage(prevRole.Value, [.. updates.SelectMany(u => u.Contents)]));
                updates.Clear();
            }
        }
    }

    private ConsoleColor GetColorForRole(ChatRole role)
    {
        if (role == ChatRole.Assistant)
        {
            return ConsoleColor.Red;
        }
        else if (role == ChatRole.Tool)
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
