"""Response generation using local or remote LLMs."""

from typing import Optional

from coderag.config import get_settings
from coderag.generation.citations import CitationParser
from coderag.generation.prompts import SYSTEM_PROMPT, build_prompt, build_no_context_response
from coderag.logging import get_logger
from coderag.models.response import Response
from coderag.models.query import Query
from coderag.retrieval.retriever import Retriever

logger = get_logger(__name__)


class ResponseGenerator:
    """Generates grounded responses using local or remote LLMs."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
    ) -> None:
        self.settings = get_settings()
        self.retriever = retriever or Retriever()
        self.citation_parser = CitationParser()

        self.provider = self.settings.models.llm_provider.lower()
        self._client = None
        self._local_model = None
        self._local_tokenizer = None

        logger.info("ResponseGenerator initialized", provider=self.provider)

    def _get_api_client(self):
        """Get or create API client for remote providers."""
        if self._client is not None:
            return self._client

        import httpx
        from openai import OpenAI

        api_key = self.settings.models.llm_api_key
        if not api_key:
            raise ValueError(f"API key required for provider: {self.provider}")

        # Provider-specific configurations
        provider_configs = {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-4o-mini",
            },
            "groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "default_model": "llama-3.3-70b-versatile",
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com/v1",
                "default_model": "claude-3-5-sonnet-20241022",
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "default_model": "anthropic/claude-3.5-sonnet",
            },
            "together": {
                "base_url": "https://api.together.xyz/v1",
                "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            },
        }

        config = provider_configs.get(self.provider, {})
        base_url = self.settings.models.llm_api_base or config.get("base_url")

        if not base_url:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Set default model if not specified and it's a known provider
        if self.settings.models.llm_name.startswith("Qwen/"):
            self.model_name = config.get("default_model", self.settings.models.llm_name)
        else:
            self.model_name = self.settings.models.llm_name

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(timeout=120.0),
        )

        logger.info("API client created", provider=self.provider, model=self.model_name)
        return self._client

    def _load_local_model(self):
        """Load local model with transformers."""
        if self._local_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Local LLM requires a CUDA-capable GPU. Options:\n"
                "  1. Use a cloud provider (free): MODEL_LLM_PROVIDER=groq\n"
                "     Get API key at: https://console.groq.com/keys\n"
                "  2. Install CUDA and a compatible GPU"
            )

        logger.info("Loading local LLM", model=self.settings.models.llm_name)

        if self.settings.models.llm_use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        self._local_tokenizer = AutoTokenizer.from_pretrained(
            self.settings.models.llm_name,
            trust_remote_code=True,
        )

        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.settings.models.llm_name,
            quantization_config=bnb_config,
            device_map=self.settings.models.llm_device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        logger.info("Local LLM loaded successfully")

    def generate(self, query: Query) -> Response:
        """Generate a response for a query."""
        # Retrieve relevant chunks
        chunks, context = self.retriever.retrieve_with_context(
            query.question,
            query.repo_id,
            query.top_k,
        )

        # Handle no results
        if not chunks:
            return Response(
                answer=build_no_context_response(),
                citations=[],
                retrieved_chunks=[],
                grounded=False,
                query_id=query.id,
            )

        # Build prompt and generate
        prompt = build_prompt(query.question, context)

        if self.provider == "local":
            answer = self._generate_local(prompt)
        else:
            answer = self._generate_api(prompt)

        # Parse citations from answer
        citations = self.citation_parser.parse_citations(answer)

        # Determine if response is grounded
        grounded = len(citations) > 0 and len(chunks) > 0

        return Response(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            grounded=grounded,
            query_id=query.id,
        )

    def _generate_api(self, prompt: str) -> str:
        """Generate using remote API."""
        client = self._get_api_client()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.settings.models.llm_max_new_tokens,
            temperature=self.settings.models.llm_temperature,
            top_p=self.settings.models.llm_top_p,
        )

        return response.choices[0].message.content.strip()

    def _generate_local(self, prompt: str) -> str:
        """Generate using local model."""
        import torch

        self._load_local_model()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = self._local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._local_tokenizer(text, return_tensors="pt").to(self._local_model.device)

        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=self.settings.models.llm_max_new_tokens,
                temperature=self.settings.models.llm_temperature,
                top_p=self.settings.models.llm_top_p,
                do_sample=True,
                pad_token_id=self._local_tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._local_tokenizer.decode(generated, skip_special_tokens=True)

        return response.strip()

    def unload(self) -> None:
        """Unload models from memory."""
        if self._local_model is not None:
            del self._local_model
            self._local_model = None
        if self._local_tokenizer is not None:
            del self._local_tokenizer
            self._local_tokenizer = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Models unloaded")
