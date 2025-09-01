using Microsoft.Extensions.AI;
using System.ComponentModel;
using System.Text;

namespace KameKodingAgent;

internal class Program
{
    const string MODEL_NAME = "qwen3:14b";

    static async Task Main(string[] args)
    {
        var program = new Program(@"D:\src\SepCsvSourceGenerator\");
        await program.Run();
    }

    private readonly string _rootPath;
    private readonly ChatOptions _options;
    private readonly IChatClient _chatClient;
    private readonly ConsoleColor _defaultForeColor;
    private readonly List<ChatMessage> _conversation;

    private Program(string rootPath)
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
            Tools =
            [
                AIFunctionFactory.Create(ListFiles),
                AIFunctionFactory.Create(ReadFile),
                AIFunctionFactory.Create(EditFile),
            ],
        };
        var client = new OllamaSharp.OllamaApiClient("http://localhost:11434", MODEL_NAME);
        _chatClient = ChatClientBuilderChatClientExtensions.AsBuilder(client).UseFunctionInvocation().Build();
        _defaultForeColor = Console.ForegroundColor;
        _conversation = new List<ChatMessage>();
    }

    void ResetConversation()
    {
        _conversation.Clear();
        _conversation.Add(new ChatMessage(ChatRole.System, "You are a programmer who edits code files based on instructions. Before editing a file, read its contents to figure out what needs to be replaced."));
    }

    async Task Run()
    {
        ResetConversation();
        while (true)
        {
            Console.WriteLine("Please enter your prompt. Enter a blank line to finish the prompt.");

            StringBuilder sb = new StringBuilder();
            while (true)
            {
                string? line = Console.ReadLine();
                if (line == null)
                {
                    Console.WriteLine("Exiting.");
                    return;
                }
                if (line.Length == 0)
                {
                    break;
                }
                sb.AppendLine(line);
            }

            string prompt = sb.ToString().Trim();
            if (prompt == "/clear")
            {
                Console.WriteLine("Cleaning context.");
                ResetConversation();
                continue;
            }
            else if (prompt == "/exit")
            {
                Console.WriteLine("Exiting");
                return;
            }

            _conversation.Add(new ChatMessage(ChatRole.User, prompt));

            var updates = new List<ChatResponseUpdate>();
            ChatRole? prevRole = null;
            await foreach (var update in _chatClient.GetStreamingResponseAsync(_conversation, _options))
            {
                updates.Add(update);
                if (!prevRole.HasValue || (prevRole.HasValue && update.Role.HasValue && prevRole.Value != update.Role.Value))
                {
                    var role = update.Role!.Value;
                    Console.WriteLine();
                    Console.ForegroundColor = GetColorForRole(role);
                    Console.Write($"{role.Value}: ");
                    Console.ForegroundColor = _defaultForeColor;
                    prevRole = role;
                }
                Console.Write(update.Text);
            }


            // TODO: add the updates to the conversation history
            break;
        }
    }

    private ConsoleColor GetColorForRole(ChatRole role)
    {
        if (role == ChatRole.Assistant)
        {
            return ConsoleColor.Red;
        }
        else if (role == ChatRole.System)
        {
            return ConsoleColor.Green;
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
        path = Path.GetFullPath(path);

        DirectoryInfo? dir;
        if (File.Exists(path))
        {
            var fi = new FileInfo(path);
            if (fi.Name[0] == '.')
            {
                throw new Exception("Trying to read dot file: " + path);
            }
            dir = new FileInfo(path).Directory;
        }
        else if (Directory.Exists(path))
        {
            dir = new DirectoryInfo(path);
        }
        else
        {
            throw new Exception("Path does not point to anything: " + path);
        }

        while (dir != null)
        {
            if (dir.FullName == _rootPath)
            {
                // Found that we are in the root directory, so it should be ok to read or write this file.
                return path;
            }
            if (dir.Name[0] == '.')
            {
                // This is for avoiding writing into the .git directory.
                throw new Exception("File path contains a directory whose name starts with a dot: " + path);
            }
            dir = dir.Parent;
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

    [Description("Edits a file, by find the value `oldStr` and replacing it with `newStr`.")]
    string EditFile(string path, string oldStr, string newStr)
    {
        path = NormalizePath(path);
        string fileContents = File.ReadAllText(path);
        string newContents = fileContents.Replace(oldStr, newStr);
        if (fileContents == newContents)
        {
            throw new Exception("Did not find `oldStr` in the file.");
        }
        File.WriteAllText(path, newContents);
        return "OK";
    }
}
