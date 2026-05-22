import os

from unittest.mock import patch
from user_soul import Persona, PersonaDecider

PERSONAS = [Persona(id="p1", name="Alice", description="test",
                    cohort="18-30", motivations=[], pain_points=[])]


def test_env_var_sets_validated_mode():
    with patch.dict(os.environ, {"DECISION_MODE": "validated"}):
        pd = PersonaDecider(PERSONAS, api_key="test")  # no mode kwarg
        assert pd.mode == "validated"


def test_kwarg_overrides_env_var():
    with patch.dict(os.environ, {"DECISION_MODE": "validated"}):
        pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
        assert pd.mode == "fast"


def test_default_mode_is_fast():
    env = {k: v for k, v in os.environ.items() if k != "DECISION_MODE"}
    with patch.dict(os.environ, env, clear=True):
        pd = PersonaDecider(PERSONAS, api_key="test")
        assert pd.mode == "fast"


def test_mode_fast_kwarg_works_without_env():
    env = {k: v for k, v in os.environ.items() if k != "DECISION_MODE"}
    with patch.dict(os.environ, env, clear=True):
        pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
        assert pd.mode == "fast"


def test_mode_validated_kwarg_works_without_env():
    env = {k: v for k, v in os.environ.items() if k != "DECISION_MODE"}
    with patch.dict(os.environ, env, clear=True):
        pd = PersonaDecider(PERSONAS, api_key="test", mode="validated")
        assert pd.mode == "validated"
