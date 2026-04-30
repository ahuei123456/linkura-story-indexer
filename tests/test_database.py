from typing import Any

from linkura_story_indexer import database
from linkura_story_indexer.database import EmbeddingDocument


def test_model_and_chroma_env_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("LINKURA_CHAT_MODEL", raising=False)
    monkeypatch.delenv("LINKURA_INGEST_PROVIDER", raising=False)
    monkeypatch.delenv("LINKURA_INGEST_MODEL", raising=False)
    monkeypatch.delenv("LINKURA_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LINKURA_CHROMA_DB_PATH", raising=False)

    assert database.get_chat_model_name() == database.DEFAULT_CHAT_MODEL
    assert database.get_generation_provider_name() == database.DEFAULT_GENERATION_PROVIDER
    assert database.get_generation_model_name() == database.DEFAULT_CHAT_MODEL
    assert database.get_embedding_model_name() == database.DEFAULT_EMBEDDING_MODEL
    assert database.get_chroma_db_path() == database.DEFAULT_CHROMA_DB_PATH

    monkeypatch.setenv("LINKURA_CHAT_MODEL", "custom-chat")
    monkeypatch.setenv("LINKURA_INGEST_PROVIDER", "openai")
    monkeypatch.setenv("LINKURA_INGEST_MODEL", "custom-ingest")
    monkeypatch.setenv("LINKURA_EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("LINKURA_CHROMA_DB_PATH", "./custom_chroma")

    assert database.get_chat_model_name() == "custom-chat"
    assert database.get_generation_provider_name() == "openai"
    assert database.get_generation_model_name() == "custom-ingest"
    assert database.get_embedding_model_name() == "custom-embedding"
    assert database.get_chroma_db_path() == "./custom_chroma"


def test_generation_model_falls_back_to_chat_model(monkeypatch):
    monkeypatch.delenv("LINKURA_INGEST_MODEL", raising=False)
    monkeypatch.setenv("LINKURA_CHAT_MODEL", "fallback-chat")

    assert database.get_generation_model_name() == "fallback-chat"


def test_generation_model_uses_google_by_default(monkeypatch):
    database.reset_client_caches()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LINKURA_INGEST_PROVIDER", raising=False)
    monkeypatch.setenv("LINKURA_INGEST_MODEL", "gemini-custom")

    class FakeGoogleProvider:
        def __init__(self, api_key: str):
            self.api_key = api_key

    class FakeGoogleModel:
        created: list[tuple[str, Any]] = []

        def __init__(self, model_name: str, *, provider: Any):
            self.model_name = model_name
            self.provider = provider
            self.created.append((model_name, provider))

    monkeypatch.setattr(database, "GoogleProvider", FakeGoogleProvider)
    monkeypatch.setattr(database, "GoogleModel", FakeGoogleModel)

    assert database.create_generation_model() is database.create_generation_model()
    assert len(FakeGoogleModel.created) == 1
    assert FakeGoogleModel.created[0][0] == "gemini-custom"


def test_generation_model_uses_openai_when_selected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LINKURA_INGEST_PROVIDER", "openai")
    monkeypatch.setenv("LINKURA_INGEST_MODEL", "gpt-5-mini")
    calls: list[str | None] = []

    def fake_create_openai_model(model_name: str | None = None) -> str:
        calls.append(model_name)
        return "openai-model"

    monkeypatch.setattr(database, "create_openai_model", fake_create_openai_model)

    assert database.create_generation_model() == "openai-model"
    assert calls == ["gpt-5-mini"]


def test_generation_settings_validate_only_selected_provider(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LINKURA_INGEST_PROVIDER", "openai")
    monkeypatch.setenv("LINKURA_INGEST_MODEL", "gpt-5-mini")

    database.initialize_generation_settings()


def test_openai_generation_requires_openai_model_when_chat_model_is_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LINKURA_INGEST_PROVIDER", "openai")
    monkeypatch.delenv("LINKURA_INGEST_MODEL", raising=False)
    monkeypatch.delenv("LINKURA_CHAT_MODEL", raising=False)

    try:
        database.initialize_generation_settings()
    except ValueError as exc:
        assert "LINKURA_INGEST_MODEL" in str(exc)
    else:
        raise AssertionError("OpenAI generation should require an OpenAI model")


def test_ingest_settings_still_require_google_for_embeddings(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LINKURA_INGEST_PROVIDER", "openai")
    monkeypatch.setenv("LINKURA_INGEST_MODEL", "gpt-5-mini")

    try:
        database.initialize_ingest_settings()
    except ValueError as exc:
        assert "GOOGLE_API_KEY" in str(exc)
    else:
        raise AssertionError("initialize_ingest_settings should require GOOGLE_API_KEY")


def test_client_and_model_helpers_return_singletons(monkeypatch):
    database.reset_client_caches()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    class FakePersistentClient:
        created: list[str] = []

        def __init__(self, path: str):
            self.path = path
            self.collections: dict[str, object] = {}
            self.created.append(path)

        def get_or_create_collection(self, name: str) -> object:
            if name not in self.collections:
                self.collections[name] = object()
            return self.collections[name]

    class FakeGenAIClient:
        created: list[str] = []

        def __init__(self, api_key: str):
            self.api_key = api_key
            self.created.append(api_key)

    class FakeGoogleProvider:
        def __init__(self, api_key: str):
            self.api_key = api_key

    class FakeGoogleModel:
        created: list[tuple[str, Any]] = []

        def __init__(self, model_name: str, *, provider: Any):
            self.model_name = model_name
            self.provider = provider
            self.created.append((model_name, provider))

    monkeypatch.setattr(database.chromadb, "PersistentClient", FakePersistentClient)
    monkeypatch.setattr(database.genai, "Client", FakeGenAIClient)
    monkeypatch.setattr(database, "GoogleProvider", FakeGoogleProvider)
    monkeypatch.setattr(database, "GoogleModel", FakeGoogleModel)

    assert database.get_chroma_client() is database.get_chroma_client()
    assert database.get_chroma_collection() is database.get_chroma_collection()
    assert database.get_genai_client() is database.get_genai_client()
    assert database.create_google_model() is database.create_google_model()

    assert FakePersistentClient.created == [database.DEFAULT_CHROMA_DB_PATH]
    assert FakeGenAIClient.created == ["test-key"]
    assert len(FakeGoogleModel.created) == 1


def test_embed_texts_batches_single_sdk_call_for_batch(monkeypatch):
    database.reset_client_caches()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("LINKURA_EMBEDDING_MODEL", "text-embedding-004")

    calls: list[dict[str, Any]] = []

    class FakeEmbedding:
        def __init__(self, values: list[float]):
            self.values = values

    class FakeResponse:
        def __init__(self, embeddings: list[FakeEmbedding]):
            self.embeddings = embeddings

    class FakeModels:
        def embed_content(self, *, model: str, contents: list[str], config: Any) -> FakeResponse:
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse(
                [FakeEmbedding([float(index)]) for index, _ in enumerate(contents)]
            )

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            self.models = FakeModels()

    monkeypatch.setattr(database.genai, "Client", FakeGenAIClient)

    vectors = database.embed_texts(["one", "two", "three"], batch_size=10)

    assert vectors == [[0.0], [1.0], [2.0]]
    assert len(calls) == 1
    assert calls[0]["model"] == "text-embedding-004"
    assert calls[0]["contents"] == ["one", "two", "three"]
    assert calls[0]["config"].task_type == database.RETRIEVAL_DOCUMENT


def test_embed_texts_respects_batch_size(monkeypatch):
    database.reset_client_caches()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("LINKURA_EMBEDDING_MODEL", "text-embedding-004")

    batch_lengths: list[int] = []

    class FakeEmbedding:
        def __init__(self, values: list[float]):
            self.values = values

    class FakeResponse:
        def __init__(self, embeddings: list[FakeEmbedding]):
            self.embeddings = embeddings

    class FakeModels:
        def embed_content(self, *, model: str, contents: list[str], config: Any) -> FakeResponse:
            batch_lengths.append(len(contents))
            return FakeResponse([FakeEmbedding([1.0]) for _ in contents])

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            self.models = FakeModels()

    monkeypatch.setattr(database.genai, "Client", FakeGenAIClient)

    database.embed_texts(["one", "two", "three"], batch_size=2)

    assert batch_lengths == [2, 1]


def test_embed_texts_uses_inline_document_format_for_gemini_embedding_2(monkeypatch):
    database.reset_client_caches()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LINKURA_EMBEDDING_MODEL", raising=False)

    calls: list[dict[str, Any]] = []

    class FakeEmbedding:
        def __init__(self, values: list[float]):
            self.values = values

    class FakeResponse:
        def __init__(self, embeddings: list[FakeEmbedding]):
            self.embeddings = embeddings

    class FakeModels:
        def embed_content(
            self,
            *,
            model: str,
            contents: list[str],
            config: Any | None = None,
        ) -> FakeResponse:
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse([FakeEmbedding([float(len(calls))])])

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            self.models = FakeModels()

    monkeypatch.setattr(database.genai, "Client", FakeGenAIClient)

    vectors = database.embed_texts(
        [
            EmbeddingDocument(text="one", title="first"),
            EmbeddingDocument(text="two", title="second"),
            "three",
        ],
        batch_size=10,
    )

    assert vectors == [[1.0], [2.0], [3.0]]
    assert calls == [
        {
            "model": database.DEFAULT_EMBEDDING_MODEL,
            "contents": ["title: first | text: one"],
            "config": None,
        },
        {
            "model": database.DEFAULT_EMBEDDING_MODEL,
            "contents": ["title: second | text: two"],
            "config": None,
        },
        {
            "model": database.DEFAULT_EMBEDDING_MODEL,
            "contents": ["title: none | text: three"],
            "config": None,
        },
    ]


def test_embed_texts_uses_inline_query_format_for_gemini_embedding_2(monkeypatch):
    database.reset_client_caches()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LINKURA_EMBEDDING_MODEL", raising=False)

    calls: list[dict[str, Any]] = []

    class FakeEmbedding:
        def __init__(self, values: list[float]):
            self.values = values

    class FakeResponse:
        def __init__(self, embeddings: list[FakeEmbedding]):
            self.embeddings = embeddings

    class FakeModels:
        def embed_content(
            self,
            *,
            model: str,
            contents: list[str],
            config: Any | None = None,
        ) -> FakeResponse:
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse([FakeEmbedding([1.0])])

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            self.models = FakeModels()

    monkeypatch.setattr(database.genai, "Client", FakeGenAIClient)

    vectors = database.embed_texts(["where is Kaho?"], task_type=database.RETRIEVAL_QUERY)

    assert vectors == [[1.0]]
    assert calls == [
        {
            "model": database.DEFAULT_EMBEDDING_MODEL,
            "contents": ["task: search result | query: where is Kaho?"],
            "config": None,
        }
    ]


def test_embed_texts_falls_back_when_batch_response_count_mismatches(monkeypatch):
    database.reset_client_caches()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("LINKURA_EMBEDDING_MODEL", "text-embedding-004")

    call_contents: list[list[str]] = []

    class FakeEmbedding:
        def __init__(self, values: list[float]):
            self.values = values

    class FakeResponse:
        def __init__(self, embeddings: list[FakeEmbedding]):
            self.embeddings = embeddings

    class FakeModels:
        def embed_content(self, *, model: str, contents: list[str], config: Any) -> FakeResponse:
            call_contents.append(contents)
            if len(contents) > 1:
                return FakeResponse([FakeEmbedding([0.0])])
            return FakeResponse([FakeEmbedding([float(len(call_contents))])])

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            self.models = FakeModels()

    monkeypatch.setattr(database.genai, "Client", FakeGenAIClient)

    vectors = database.embed_texts(["one", "two"], batch_size=10)

    assert vectors == [[2.0], [3.0]]
    assert call_contents == [["one", "two"], ["one"], ["two"]]
