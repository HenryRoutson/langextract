"""Provider implementation for Azure OpenAI."""

import os
import concurrent.futures
from typing import Any, Iterator, Sequence

import langextract as lx


@lx.providers.registry.register(r'^azureOpenAI', priority=10)
class AzureOpenAILanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for Azure OpenAI.

    This provider handles model IDs matching: ['^azureOpenAI']
    """

    def __init__(
        self,
        model_id: str = 'gpt-35-turbo',
        azure_deployment: str | None = None,
        api_version: str = '2023-06-01-preview',
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        format_type: lx.data.FormatType = lx.data.FormatType.JSON,
        temperature: float | None = None,
        max_workers: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI provider.

        Args:
            model_id: The model identifier (e.g., 'azureOpenAI-gpt-35-turbo').
            azure_deployment: Azure deployment name.
            api_version: Azure API version.
            azure_endpoint: Azure OpenAI endpoint URL.
            api_key: Azure OpenAI API key.
            format_type: Output format (JSON or YAML).
            temperature: Sampling temperature.
            max_workers: Maximum number of parallel API calls.
            **kwargs: Additional provider-specific parameters.
        """
        # AI : Lazy import to avoid import errors if langchain-openai is not installed
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI provider requires langchain-openai package. '
                'Install with: pip install langchain-openai'
            ) from e

        super().__init__() #type:ignore
        
        self.model_id = model_id
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT')
        self.api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        self.format_type = format_type
        self.temperature = temperature
        self.max_workers = max_workers

        if not self.api_key:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY environment variable.'
            )

        if not self.azure_endpoint:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT environment variable.'
            )

        # AI : Initialize the Azure OpenAI client using langchain
        # AI : Set environment variables for langchain to pick up automatically
        if self.api_key:
            os.environ['AZURE_OPENAI_API_KEY'] = self.api_key
        if self.azure_endpoint:
            os.environ['AZURE_OPENAI_ENDPOINT'] = self.azure_endpoint
            
        self._client = AzureChatOpenAI(
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
            temperature=self.temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self._extra_kwargs = kwargs or {}


    def _process_single_prompt(self, prompt: str, config: dict[str, Any]) -> lx.inference.ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            # AI : Create messages in LangChain format
            messages = [
                (
                    "system",
                    "You are a helpful assistant that responds in JSON format."
                    if self.format_type == lx.data.FormatType.JSON
                    else "You are a helpful assistant that responds in YAML format."
                ),
                ("human", prompt),
            ]

            # AI : Use LangChain's invoke method
            ai_msg = self._client.invoke(messages)
            output_text : str = str(ai_msg.content) if ai_msg.content else "" # type:ignore

            return lx.inference.ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            raise lx.exceptions.InferenceRuntimeError(
                f'Azure OpenAI API error: {str(e)}', original=e
            ) from e

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[lx.inference.ScoredOutput]]:
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters.

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
        # AI : Simple merge of kwargs
        merged_kwargs = dict(self._extra_kwargs or {})
        merged_kwargs.update(kwargs)

        config: dict[str, Any] = {}
        # AI : Extract temperature if provided
        temp = merged_kwargs.get('temperature', self.temperature)
        if temp is not None:
            config['temperature'] = temp
            # AI : Update client temperature for this batch
            self._client.temperature = temp

        # AI : Use parallel processing for batches larger than 1
        if len(batch_prompts) > 1 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(batch_prompts))
            ) as executor:
                future_to_index = {
                    executor.submit(
                        self._process_single_prompt, prompt, config.copy()
                    ): i
                    for i, prompt in enumerate(batch_prompts)
                }

                results: list[lx.inference.ScoredOutput | None] = [None] * len(
                    batch_prompts
                )
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        raise lx.exceptions.InferenceRuntimeError(
                            f'Parallel inference error: {str(e)}', original=e
                        ) from e

                for result in results:
                    if result is None:
                        raise lx.exceptions.InferenceRuntimeError(
                            'Failed to process one or more prompts'
                        )
                    yield [result]
        else:
            # AI : Sequential processing for single prompt or worker
            for prompt in batch_prompts:
                result = self._process_single_prompt(prompt, config.copy())
                yield [result]
