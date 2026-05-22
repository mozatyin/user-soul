# mcv/tests/test_bridge_smoke.py

def test_eltm_bridge_types_importable():
    """All three new types are importable from mcv root."""
    from user_soul import MCVClient, FeatureAAR, CoherenceReport
    assert MCVClient is not None
    assert FeatureAAR is not None
    assert CoherenceReport is not None


def test_eltm_bridge_methods_exist():
    """MCVClient exposes all three ELTM bridge methods."""
    from user_soul import MCVClient
    assert hasattr(MCVClient, "research_aarrr")
    assert hasattr(MCVClient, "validate_coherence")
    assert hasattr(MCVClient, "attribute_frictions")
