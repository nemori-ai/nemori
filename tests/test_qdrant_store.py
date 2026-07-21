from unittest.mock import patch

from nemori.db.qdrant_store import QdrantVectorStore, _is_loopback_url


def test_loopback_qdrant_urls_are_detected() -> None:
    assert _is_loopback_url("localhost") is True
    assert _is_loopback_url("127.0.0.1") is True
    assert _is_loopback_url("http://127.0.0.1:16333") is True
    assert _is_loopback_url("http://[::1]:6333") is True
    assert _is_loopback_url("https://qdrant.example.test") is False


def test_local_qdrant_client_bypasses_environment_proxy() -> None:
    with patch("nemori.db.qdrant_store.QdrantClient") as client_class:
        QdrantVectorStore(url="127.0.0.1", port=16333)

    client_class.assert_called_once_with(
        url="127.0.0.1",
        port=16333,
        api_key=None,
        check_compatibility=False,
        trust_env=False,
    )


def test_remote_qdrant_client_keeps_environment_proxy_support() -> None:
    with patch("nemori.db.qdrant_store.QdrantClient") as client_class:
        QdrantVectorStore(url="https://qdrant.example.test", port=443)

    client_class.assert_called_once_with(
        url="https://qdrant.example.test",
        port=443,
        api_key=None,
        check_compatibility=False,
    )
