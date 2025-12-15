"""Response generation using Qwen2.5-Coder."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from coderag.config import get_settings
from coderag.generation.citations import CitationParser
from coderag.generation.prompts import SYSTEM_PROMPT, build_prompt, build_no_context_response
from coderag.logging import get_logger
from coderag.models.response import Citation, Query, Response, RetrievedChunk
from coderag.retrieval.retriever import Retriever

logger = get_logger(__name__)


class ResponseGenerator:
    """Generates grounded responses using Qwen2.5-Coder."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        model_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.retriever = retriever or Retriever()
        self.model_name = model_name or settings.models.llm_name
        self.max_new_tokens = settings.models.llm_max_new_tokens
        self.temperature = settings.models.llm_temperature
        self.top_p = settings.models.llm_top_p
        self.use_4bit = settings.models.llm_use_4bit

        self._model = None
        self._tokenizer = None
        self.citation_parser = CitationParser()

    def _load_model(self) -> None:
        logger.info("Loading LLM", model=self.model_name, use_4bit=self.use_4bit)

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        logger.info("LLM loaded successfully")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def generate(self, query: Query) -> Response:
        """Generate a response for a query.

        Args:
            query: The user's query

        Returns:
            Generated response with citations
        """
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
        answer = self._generate_text(prompt)

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

    def _generate_text(self, prompt: str) -> str:
        """Generate text using the LLM."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return response.strip()

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LLM unloaded")
