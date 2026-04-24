import os
from collections.abc import Sequence
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
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_CHROMA_DB_PATH = "./chroma_db"
RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
RETRIEVAL_QUERY = "RETRIEVAL_QUERY"

_chroma_clients: dict[str, Any] = {}
_chroma_collections: dict[tuple[str, str], Any] = {}
_genai_clients: dict[str, genai.Client] = {}
_google_models: dict[tuple[str, str], GoogleModel] = {}


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    return api_key


def get_chat_model_name() -> str:
    return os.getenv("LINKURA_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def get_embedding_model_name() -> str:
    return os.getenv("LINKURA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_chroma_db_path() -> str:
    return os.getenv("LINKURA_CHROMA_DB_PATH", DEFAULT_CHROMA_DB_PATH)


def initialize_settings() -> None:
    """Validates environment configuration for commands that call Google APIs."""
    get_google_api_key()


def create_text_agent(instructions: str) -> Agent[None, str]:
    """Creates a PydanticAI agent backed by Gemini."""
    return Agent(create_google_model(), instructions=instructions)


def create_google_model() -> GoogleModel:
    api_key = get_google_api_key()
    model_name = get_chat_model_name()
    cache_key = (api_key, model_name)
    if cache_key not in _google_models:
        _google_models[cache_key] = GoogleModel(
            model_name,
            provider=GoogleProvider(api_key=api_key),
        )
    return _google_models[cache_key]


def get_genai_client() -> genai.Client:
    api_key = get_google_api_key()
    if api_key not in _genai_clients:
        _genai_clients[api_key] = genai.Client(api_key=api_key)
    return _genai_clients[api_key]


def _supports_batch_embeddings(model_name: str) -> bool:
    # google-genai special-cases gemini-embedding-2 by normalizing a list of
    # strings into one Content, so it does not produce one embedding per string.
    return "gemini-embedding-2" not in model_name


def _embed_text_batch(client: genai.Client, texts: Sequence[str], model_name: str, task_type: str) -> list[list[float]]:
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
    texts: Sequence[str],
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
        batch = list(texts[start : start + effective_batch_size])
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
