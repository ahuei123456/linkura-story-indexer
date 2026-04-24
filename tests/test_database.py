from typing import Any

from linkura_story_indexer import database


def test_model_and_chroma_env_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("LINKURA_CHAT_MODEL", raising=False)
    monkeypatch.delenv("LINKURA_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LINKURA_CHROMA_DB_PATH", raising=False)

    assert database.get_chat_model_name() == database.DEFAULT_CHAT_MODEL
    assert database.get_embedding_model_name() == database.DEFAULT_EMBEDDING_MODEL
    assert database.get_chroma_db_path() == database.DEFAULT_CHROMA_DB_PATH

    monkeypatch.setenv("LINKURA_CHAT_MODEL", "custom-chat")
    monkeypatch.setenv("LINKURA_EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("LINKURA_CHROMA_DB_PATH", "./custom_chroma")

    assert database.get_chat_model_name() == "custom-chat"
    assert database.get_embedding_model_name() == "custom-embedding"
    assert database.get_chroma_db_path() == "./custom_chroma"


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


def test_embed_texts_uses_single_item_calls_for_gemini_embedding_2(monkeypatch):
    database.reset_client_caches()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LINKURA_EMBEDDING_MODEL", raising=False)

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
            return FakeResponse([FakeEmbedding([float(len(call_contents))])])

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            self.models = FakeModels()

    monkeypatch.setattr(database.genai, "Client", FakeGenAIClient)

    vectors = database.embed_texts(["one", "two", "three"], batch_size=10)

    assert vectors == [[1.0], [2.0], [3.0]]
    assert call_contents == [["one"], ["two"], ["three"]]


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
