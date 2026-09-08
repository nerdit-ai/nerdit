"""Serve model backends through a common OpenAI-compatible env contract."""

from nerdit.core.models.backend import (
    DEFAULT_OLLAMA_IMAGE,
    DEFAULT_PULL_TIMEOUT_S,
    DEFAULT_VLLM_IMAGE,
    OLLAMA_PORT,
    VLLM_PORT,
    ModelBackend,
    ModelPullError,
    ModelServerUnreachableError,
    OllamaBackend,
    VllmBackend,
    sanitize_model_name,
)
from nerdit.core.models.binding import (
    LOCAL_MODEL_API_KEY,
    BindingNotReady,
    ResolvedBinding,
    inject_env,
    resolve_binding,
    resolve_bindings,
)
from nerdit.core.models.controller import ModelController

__all__ = [
    "DEFAULT_OLLAMA_IMAGE",
    "DEFAULT_PULL_TIMEOUT_S",
    "DEFAULT_VLLM_IMAGE",
    "LOCAL_MODEL_API_KEY",
    "OLLAMA_PORT",
    "VLLM_PORT",
    "BindingNotReady",
    "ModelBackend",
    "ModelController",
    "ModelPullError",
    "ModelServerUnreachableError",
    "OllamaBackend",
    "VllmBackend",
    "ResolvedBinding",
    "inject_env",
    "resolve_binding",
    "resolve_bindings",
    "sanitize_model_name",
]
