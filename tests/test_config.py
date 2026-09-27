from config import Config


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("RAG_FINAL_K", "3")
    monkeypatch.setenv("RAG_STRATEGY", "semantic")
    monkeypatch.setenv("RAG_API_KEY", "secret")

    config = Config(_env_file=None)

    assert config.final_k == 3
    assert config.strategy == "semantic"
    assert config.api_key.get_secret_value() == "secret"
    assert "secret" not in repr(config)  # SecretStr keeps keys out of logs/tracebacks


def test_defaults_without_env():
    config = Config(_env_file=None)

    assert config.strategy == "recursive"
    assert config.api_key is None


def test_index_paths_use_legacy_layout_before_first_versioned_build(tmp_path):
    config = Config(_env_file=None, data_dir=str(tmp_path), strategy="fixed")

    assert config.current_index_dir() is None
    assert config.chroma_persist_dir == str(tmp_path / "chroma_fixed")
    assert config.bm25_path == str(tmp_path / "bm25_fixed.pkl")


def test_index_paths_follow_current_pointer(tmp_path):
    config = Config(_env_file=None, data_dir=str(tmp_path), strategy="fixed")
    config.current_index_pointer.write_text("v2\n")

    version = tmp_path / "indexes" / "fixed" / "v2"
    assert config.chroma_persist_dir == str(version / "chroma")
    assert config.bm25_path == str(version / "bm25.pkl")
