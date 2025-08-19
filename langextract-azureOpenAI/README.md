        # LangExtract Azure OpenAI Provider

A provider plugin for LangExtract that supports Azure OpenAI models.

## Installation

```bash
pip install -e .
```

## Supported Model IDs

- `azureOpenAI*`: Models matching pattern ^azureOpenAI

## Environment Variables

- `AZURE_OPENAI_API_KEY`: API key for authentication
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL

## Usage

```python
import langextract as lx

result = lx.extract(
    text="Your document here",
    model_id="azureOpenAI-gpt-35-turbo",
    prompt_description="Extract entities",
    examples=[...]
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## License

Apache License 2.0
