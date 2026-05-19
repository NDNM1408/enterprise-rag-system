"""
Unit tests for WorkerContainer (composition root).

Verifies that:
- init() is idempotent (called multiple times is safe).
- Properties trigger lazy init when accessed before explicit init().
- Dependencies are not re-created between property accesses (same object).
- The SQLAlchemy engine is created with NullPool.
"""

from unittest.mock import MagicMock, patch, call


class TestWorkerContainer:

    def _fresh_container(self):
        """Return a new, uninitialized WorkerContainer."""
        from app.container import WorkerContainer
        return WorkerContainer()

    # ------------------------------------------------------------------
    # init() behaviour
    # ------------------------------------------------------------------

    @patch("app.container.S3ClientService")
    @patch("app.container.MarkdownSplitter")
    @patch("app.container.DocumentParser")
    @patch("app.container.EmbeddingService")
    @patch("app.container.async_sessionmaker")
    @patch("app.container.create_async_engine")
    def test_init_sets_initialized_flag(
        self, mock_engine, mock_sm, mock_emb, mock_parser, mock_splitter, mock_s3
    ):
        c = self._fresh_container()
        assert not c._initialized
        c.init()
        assert c._initialized

    @patch("app.container.S3ClientService")
    @patch("app.container.MarkdownSplitter")
    @patch("app.container.DocumentParser")
    @patch("app.container.EmbeddingService")
    @patch("app.container.async_sessionmaker")
    @patch("app.container.create_async_engine")
    def test_init_is_idempotent(
        self, mock_engine, mock_sm, mock_emb, mock_parser, mock_splitter, mock_s3
    ):
        """Calling init() twice must not create resources a second time."""
        c = self._fresh_container()
        c.init()
        c.init()
        mock_engine.assert_called_once()
        mock_s3.assert_called_once()

    @patch("app.container.S3ClientService")
    @patch("app.container.MarkdownSplitter")
    @patch("app.container.DocumentParser")
    @patch("app.container.EmbeddingService")
    @patch("app.container.async_sessionmaker")
    @patch("app.container.create_async_engine")
    def test_properties_return_same_object(
        self, mock_engine, mock_sm, mock_emb, mock_parser, mock_splitter, mock_s3
    ):
        """Accessing a property twice returns the identical object."""
        c = self._fresh_container()
        c.init()
        assert c.session_factory is c.session_factory
        assert c.s3 is c.s3
        assert c.parser is c.parser
        assert c.splitter is c.splitter
        assert c.embedding_service is c.embedding_service

    @patch("app.container.S3ClientService")
    @patch("app.container.MarkdownSplitter")
    @patch("app.container.DocumentParser")
    @patch("app.container.EmbeddingService")
    @patch("app.container.async_sessionmaker")
    @patch("app.container.create_async_engine")
    def test_lazy_init_on_property_access(
        self, mock_engine, mock_sm, mock_emb, mock_parser, mock_splitter, mock_s3
    ):
        """Accessing a property before init() triggers init automatically."""
        c = self._fresh_container()
        assert not c._initialized
        _ = c.session_factory  # triggers lazy init
        assert c._initialized

    @patch("app.container.S3ClientService")
    @patch("app.container.MarkdownSplitter")
    @patch("app.container.DocumentParser")
    @patch("app.container.EmbeddingService")
    @patch("app.container.async_sessionmaker")
    @patch("app.container.create_async_engine")
    def test_engine_uses_nullpool(
        self, mock_engine, mock_sm, mock_emb, mock_parser, mock_splitter, mock_s3
    ):
        """Engine must be created with NullPool so it is event-loop-agnostic."""
        from sqlalchemy.pool import NullPool
        c = self._fresh_container()
        c.init()
        _, kwargs = mock_engine.call_args
        assert kwargs.get("poolclass") is NullPool

    @patch("app.container.S3ClientService")
    @patch("app.container.MarkdownSplitter")
    @patch("app.container.DocumentParser")
    @patch("app.container.EmbeddingService")
    @patch("app.container.async_sessionmaker")
    @patch("app.container.create_async_engine")
    def test_splitter_uses_settings(
        self, mock_engine, mock_sm, mock_emb, mock_parser, mock_splitter, mock_s3
    ):
        """MarkdownSplitter must receive values from settings."""
        from app.configurations.configurations import settings
        c = self._fresh_container()
        c.init()
        mock_splitter.assert_called_once_with(
            tokenizer_model=settings.TIKTOKEN_MODEL_NAME,
            retrieve_max_tokens=settings.RETRIEVE_MAX_TOKENS,
            retrieve_target_tokens=settings.RETRIEVE_TARGET_TOKENS,
        )
