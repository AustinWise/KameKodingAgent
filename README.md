# KameKodingAgent

A simple LLM coding agent written in C# that supports a variety of different LLM APIs.
In principal any LLM API that has an implementation of
[Microsoft.Extensions.AI](https://learn.microsoft.com/en-us/dotnet/ai/microsoft-extensions-ai)'s
[`IChatClient` interface](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.ai.ichatclient)
should work. Currently the following are implemented:

* Google Cloud Platform's Vertex AI - `--llm-backend=VertexAi`
* Anthropic - `--llm-backend=Anthropic`
* Ollama - `--llm-backend=Ollama`
* OpenAI's responses API, the newest OpenAI API - `--llm-backend=OpenAiResponses`
* OpenAI's chat API, a widely implemented API - `--lmm-backup=OpenAiChat`

## Using with llama.cpp

If you have started llama.cpp with `llama server`, you can this program with:

```bash
dotnet run -- --llm-backend OpenAiChat --root-directory ~/some-project/ --endpoint http://127.0.0.1:8080
```

## Pronunciation

Kame is pronounced "kah-may". It means turtle in Japanese.
