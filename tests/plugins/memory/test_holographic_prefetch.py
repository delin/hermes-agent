from unittest.mock import MagicMock

from plugins.memory.holographic import HolographicMemoryProvider


def _candidate(content: str, tags: str = "") -> dict:
    return {
        "content": content,
        "tags": tags,
        "trust_score": 0.5,
        "score": 0.5,
    }


def test_prefetch_requires_lexical_coverage_and_caps_results():
    provider = HolographicMemoryProvider(
        {"prefetch_limit": 2, "prefetch_min_token_overlap": 2}
    )
    retriever = MagicMock()
    retriever._tokenize.side_effect = lambda text: {
        token.lower().strip(".,:;()[]")
        for token in text.split()
        if token.strip(".,:;()[]")
    }
    retriever.search.return_value = [
        _candidate("Hermes profile boot hook"),
        _candidate("Hermes compression patch preserves active request"),
        _candidate("Compression patch bounds summary context"),
        _candidate("Hermes MCP kube debug"),
    ]
    provider._retriever = retriever

    block = provider.prefetch("Hermes compression patch")

    assert "profile boot hook" not in block
    assert "MCP kube debug" not in block
    assert "Hermes compression patch preserves active request" in block
    assert "Compression patch bounds summary context" in block
    assert block.count("\n-") == 2
    assert "[0.5]" not in block


def test_prefetch_real_store_applies_lexical_gate(tmp_path):
    provider = HolographicMemoryProvider(
        {
            "db_path": str(tmp_path / "memory_store.db"),
            "hrr_dim": 64,
            "prefetch_limit": 2,
            "prefetch_min_token_overlap": 2,
        }
    )
    provider.initialize(session_id="prefetch-integration")
    try:
        provider._store.add_fact("Hermes profile boot hook")
        provider._store.add_fact(
            "Hermes compression patch preserves active request"
        )
        provider._store.add_fact("Compression patch bounds summary context")

        block = provider.prefetch("Hermes compression patch")

        assert "profile boot hook" not in block
        assert "Hermes compression patch preserves active request" in block
        assert "Compression patch bounds summary context" in block
    finally:
        provider.shutdown()


def test_prefetch_single_token_query_still_allows_exact_match():
    provider = HolographicMemoryProvider(
        {"prefetch_limit": 2, "prefetch_min_token_overlap": 2}
    )
    retriever = MagicMock()
    retriever._tokenize.side_effect = lambda text: {token.lower() for token in text.split()}
    retriever.search.return_value = [_candidate("Arcadia context")]
    provider._retriever = retriever

    assert "Arcadia context" in provider.prefetch("Arcadia")


def test_prefetch_can_be_disabled():
    provider = HolographicMemoryProvider({"prefetch_limit": 0})
    provider._retriever = MagicMock()

    assert provider.prefetch("anything") == ""
    provider._retriever.search.assert_not_called()
