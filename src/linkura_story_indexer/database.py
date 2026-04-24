import os
from collections.abc import Sequence
from typing import Any, cast

import chromadb
from dotenv import load_dotenv
from google import genai
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

load_dotenv()

CHAT_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "gemini-embedding-2"
CHROMA_DB_PATH = "./chroma_db"


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    return api_key


def initialize_settings() -> None:
    """Validates environment configuration for commands that call Google APIs."""
    get_google_api_key()


def create_text_agent(instructions: str) -> Agent[None, str]:
    """Creates a PydanticAI agent backed by Gemini."""
    return Agent(create_google_model(), instructions=instructions)


def create_google_model() -> GoogleModel:
    return GoogleModel(CHAT_MODEL, provider=GoogleProvider(api_key=get_google_api_key()))


def get_genai_client() -> genai.Client:
    return genai.Client(api_key=get_google_api_key())


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    client = get_genai_client()
    vectors = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=cast(Any, text),
        )
        embeddings = response.embeddings or []
        if not embeddings:
            raise ValueError("Google embedding response did not include embeddings")
        values = cast(list[float] | None, embeddings[0].values)
        if values is None:
            raise ValueError("Google embedding response did not include vector values")
        vectors.append(values)
    return vectors


def get_chroma_collection(collection_name: str = "story_nodes") -> Any:
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return db.get_or_create_collection(collection_name)
