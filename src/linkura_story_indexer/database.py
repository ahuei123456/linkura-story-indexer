import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

load_dotenv()

DEFAULT_CHAT_MODEL = "gemini-3-flash-preview"
DEFAULT_GENERATION_PROVIDER = "google"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_CHROMA_DB_PATH = "./chroma_db"
RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
SUPPORTED_GENERATION_PROVIDERS = {"google", "openai"}


@dataclass(frozen=True)
class EmbeddingDocument:
    text: str
    title: str = "none"


EmbeddingInput = str | EmbeddingDocument

_chroma_clients: dict[str, Any] = {}
_chroma_collections: dict[tuple[str, str], Any] = {}
_genai_clients: dict[str, genai.Client] = {}
_google_models: dict[tuple[str, str], GoogleModel] = {}
_openai_models: dict[tuple[str, str, str], Any] = {}


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    return api_key


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return api_key


def get_chat_model_name() -> str:
    return os.getenv("LINKURA_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def get_generation_provider_name() -> str:
    return os.getenv("LINKURA_INGEST_PROVIDER", DEFAULT_GENERATION_PROVIDER).strip().lower()


def get_generation_model_name() -> str:
    return os.getenv("LINKURA_INGEST_MODEL") or get_chat_model_name()


def get_embedding_model_name() -> str:
    return os.getenv("LINKURA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_chroma_db_path() -> str:
    return os.getenv("LINKURA_CHROMA_DB_PATH", DEFAULT_CHROMA_DB_PATH)


def initialize_settings() -> None:
    """Validates environment configuration for commands that call Google APIs."""
    get_google_api_key()


def initialize_generation_settings() -> None:
    """Validates environment configuration for commands that call generation APIs."""
    provider = get_generation_provider_name()
    if provider == "google":
        get_google_api_key()
    elif provider == "openai":
        get_openai_api_key()
        if not os.getenv("LINKURA_INGEST_MODEL") and not os.getenv("LINKURA_CHAT_MODEL"):
            raise ValueError(
                "LINKURA_INGEST_PROVIDER=openai requires LINKURA_INGEST_MODEL, "
                "or LINKURA_CHAT_MODEL set to an OpenAI model."
            )
    else:
        raise ValueError(
            f"Unsupported LINKURA_INGEST_PROVIDER {provider!r}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_GENERATION_PROVIDERS))}"
        )


def initialize_ingest_settings() -> None:
    """Validates environment configuration for ingest embeddings and generation."""
    get_google_api_key()
    initialize_generation_settings()


def create_text_agent(instructions: str) -> Agent[None, str]:
    """Creates a PydanticAI agent backed by Gemini."""
    return Agent(create_google_model(), instructions=instructions)


def create_generation_text_agent(instructions: str) -> Agent[None, str]:
    """Creates a PydanticAI text agent backed by the configured generation provider."""
    return Agent(create_generation_model(), instructions=instructions)


def create_generation_model() -> Any:
    provider = get_generation_provider_name()
    model_name = get_generation_model_name()
    if provider == "google":
        return create_google_model(model_name)
    if provider == "openai":
        return create_openai_model(model_name)
    raise ValueError(
        f"Unsupported LINKURA_INGEST_PROVIDER {provider!r}. "
        f"Expected one of: {', '.join(sorted(SUPPORTED_GENERATION_PROVIDERS))}"
    )


def create_google_model(model_name: str | None = None) -> GoogleModel:
    api_key = get_google_api_key()
    model_name = model_name or get_chat_model_name()
    cache_key = (api_key, model_name)
    if cache_key not in _google_models:
        _google_models[cache_key] = GoogleModel(
            model_name,
            provider=GoogleProvider(api_key=api_key),
        )
    return _google_models[cache_key]


def create_openai_model(model_name: str | None = None) -> Any:
    api_key = get_openai_api_key()
    model_name = model_name or get_generation_model_name()
    base_url = os.getenv("OPENAI_BASE_URL", "")
    cache_key = (api_key, model_name, base_url)
    if cache_key not in _openai_models:
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError as exc:
            raise ImportError(
                "OpenAI generation requires the OpenAI extra. "
                "Install dependencies with `uv sync` after adding "
                '`pydantic-ai-slim[google,openai]`.'
            ) from exc

        _openai_models[cache_key] = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url or None, api_key=api_key),
        )
    return _openai_models[cache_key]


def get_genai_client() -> genai.Client:
    api_key = get_google_api_key()
    if api_key not in _genai_clients:
        _genai_clients[api_key] = genai.Client(api_key=api_key)
    return _genai_clients[api_key]


def _supports_batch_embeddings(model_name: str) -> bool:
    # google-genai special-cases gemini-embedding-2 by normalizing a list of
    # strings into one Content, so it does not produce one embedding per string.
    return not _uses_inline_embedding_instructions(model_name)


def _uses_inline_embedding_instructions(model_name: str) -> bool:
    return "gemini-embedding-2" in model_name


def _embedding_input_text(text: EmbeddingInput) -> str:
    if isinstance(text, EmbeddingDocument):
        return text.text
    return text


def _embedding_document_title(text: EmbeddingInput) -> str:
    if isinstance(text, EmbeddingDocument) and text.title:
        return text.title
    return "none"


def _format_embedding_content(
    text: EmbeddingInput,
    *,
    model_name: str,
    task_type: str,
) -> str:
    content = _embedding_input_text(text)
    if not _uses_inline_embedding_instructions(model_name):
        return content
    if task_type == RETRIEVAL_QUERY:
        return f"task: search result | query: {content}"
    if task_type == RETRIEVAL_DOCUMENT:
        return f"title: {_embedding_document_title(text)} | text: {content}"
    return content


def _embed_text_batch(
    client: genai.Client,
    texts: Sequence[str],
    model_name: str,
    task_type: str,
) -> list[list[float]]:
    if _uses_inline_embedding_instructions(model_name):
        response = client.models.embed_content(
            model=model_name,
            contents=cast(Any, list(texts)),
        )
    else:
        response = client.models.embed_content(
            model=model_name,
            contents=cast(Any, list(texts)),
            config=types.EmbedContentConfig(task_type=task_type),
        )
    embeddings = response.embeddings or []
    if not embeddings:
        raise ValueError("Google embedding response did not include embeddings")
    if len(embeddings) != len(texts):
        raise ValueError("Google embedding response did not match input batch size")

    vectors = []
    for embedding in embeddings:
        values = cast(list[float] | None, embedding.values)
        if values is None:
            raise ValueError("Google embedding response did not include vector values")
        vectors.append(values)
    return vectors


def embed_texts(
    texts: Sequence[EmbeddingInput],
    *,
    task_type: str = RETRIEVAL_DOCUMENT,
    batch_size: int = 32,
) -> list[list[float]]:
    if not texts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    client = get_genai_client()
    model_name = get_embedding_model_name()
    effective_batch_size = batch_size if _supports_batch_embeddings(model_name) else 1
    vectors = []
    for start in range(0, len(texts), effective_batch_size):
        batch = [
            _format_embedding_content(text, model_name=model_name, task_type=task_type)
            for text in texts[start : start + effective_batch_size]
        ]
        try:
            vectors.extend(_embed_text_batch(client, batch, model_name, task_type))
        except ValueError:
            if len(batch) == 1:
                raise
            for text in batch:
                vectors.extend(_embed_text_batch(client, [text], model_name, task_type))
    return vectors


def get_chroma_client() -> Any:
    path = get_chroma_db_path()
    if path not in _chroma_clients:
        _chroma_clients[path] = chromadb.PersistentClient(path=path)
    return _chroma_clients[path]


def get_chroma_collection(collection_name: str = "story_nodes") -> Any:
    path = get_chroma_db_path()
    cache_key = (path, collection_name)
    if cache_key not in _chroma_collections:
        _chroma_collections[cache_key] = get_chroma_client().get_or_create_collection(collection_name)
    return _chroma_collections[cache_key]


def reset_client_caches() -> None:
    """Clears cached clients and models, primarily for tests."""
    _chroma_clients.clear()
    _chroma_collections.clear()
    _genai_clients.clear()
    _google_models.clear()
    _openai_models.clear()
