"""Provider/profile configuration service."""

from coderag.config import ModelSettings, get_settings


class ProviderConfigService:
    """Expose model/profile capability flags without constructing providers."""

    def __init__(self, model_settings: ModelSettings | None = None) -> None:
        self.model_settings = model_settings or get_settings().models

    @property
    def profile(self) -> str:
        return self.model_settings.profile

    @property
    def requires_generation_provider(self) -> bool:
        return self.profile != "retrieval-only"

    @property
    def requires_local_llm(self) -> bool:
        return self.profile == "private"

    @property
    def uses_cloud_generation(self) -> bool:
        return self.requires_generation_provider and self.model_settings.llm_provider.lower() != "local"

    def generation_block_reason(self) -> str | None:
        """Return why LLM generation is disallowed for the current profile."""
        if self.profile == "retrieval-only":
            return "LLM generation is disabled by the retrieval-only profile"
        if self.profile == "private" and self.model_settings.llm_provider.lower() != "local":
            return "Private profile requires MODEL_LLM_PROVIDER=local"
        return None
