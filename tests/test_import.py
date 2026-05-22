import dataclasses
import inspect


def test_import_symbols():
    from user_soul import PersonaDecider, DecisionResult, Persona, load_or_generate
    assert PersonaDecider is not None, "PersonaDecider not importable"
    assert DecisionResult is not None, "DecisionResult not importable"
    assert Persona is not None, "Persona not importable"
    assert load_or_generate is not None, "load_or_generate not importable"


def test_persona_fields():
    from user_soul import Persona
    fields = {f.name for f in dataclasses.fields(Persona)}
    assert fields == {"id", "name", "description", "cohort", "motivations", "pain_points"}, \
        f"Persona fields mismatch: {fields}"


def test_decision_result_fields():
    from user_soul import DecisionResult
    fields = {f.name for f in dataclasses.fields(DecisionResult)}
    assert {"value", "confidence", "distribution", "mode", "tokens_used", "raw_votes"} <= fields, \
        f"DecisionResult missing fields: {fields}"


def test_persona_decider_signature():
    from user_soul import PersonaDecider
    sig = inspect.signature(PersonaDecider.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    assert "personas" in params, "PersonaDecider missing 'personas' param"
    assert "api_key" in params, "PersonaDecider missing 'api_key' param"
    assert "mode" in params, "PersonaDecider missing 'mode' param"


def test_mcvclient_importable():
    from user_soul import MCVClient
    assert MCVClient is not None


def test_compare_report_importable():
    from user_soul import CompareReport
    assert CompareReport is not None


def test_build_domain_config_importable():
    from user_soul import build_domain_config
    assert callable(build_domain_config)
