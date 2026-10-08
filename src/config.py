"""
Central configuration for the pipeline. Every tunable knob lives here and is
read once from environment variables (loaded from .env via python-dotenv).
Nothing else in the codebase should call os.environ directly -- import from
here instead, so the whole project has one source of truth for settings.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env(name: str, default: str) -> str:
    """Like os.getenv, but treats a blank/whitespace-only value (e.g. a bare
    'GENERATOR_MODEL=' line left empty in .env) the same as unset, falling
    back to `default` instead of returning an empty string."""
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    return val


@dataclass(frozen=True)
class Settings:
    # Providers
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")

    # API keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Models
    # Defaults switch automatically to Groq-hosted models when
    # LLM_PROVIDER=groq and the user hasn't overridden GENERATOR_MODEL /
    # GRADER_MODEL in .env (gpt-4o-mini would otherwise be sent to Groq,
    # which doesn't host it, and the call would fail).
    generator_model: str = _env(
        "GENERATOR_MODEL",
        "llama-3.3-70b-versatile" if os.getenv("LLM_PROVIDER") == "groq" else "gpt-4o-mini",
    )
    grader_model: str = _env(
        "GRADER_MODEL",
        "llama-3.3-70b-versatile" if os.getenv("LLM_PROVIDER") == "groq" else "gpt-4o-mini",
    )
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )

    # Chunking / retrieval
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))

    # Self-correction bounds
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    # Vector store
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "self_corrective_rag"
    )

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.tavily_api_key)

    def validate(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Either set it in .env, or set "
                "LLM_PROVIDER=local and EMBEDDING_PROVIDER=local to run "
                "without an API key."
            )
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to .env (get one free at "
                "https://console.groq.com/keys)."
            )
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise EnvironmentError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Groq does not offer an embeddings API, so if you're using "
                "LLM_PROVIDER=groq, set EMBEDDING_PROVIDER=local in .env "
                "instead (free, runs locally, no key needed)."
            )


settings = Settings()
