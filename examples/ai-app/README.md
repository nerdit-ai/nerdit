# ai-app — the P5 north-star fixture

A minimal, dependency-free Node app used to demonstrate (and manually verify)
the Nerdit AI wedge end-to-end: it is built by the Node buildpack, gets a URL,
and talks to **both** kinds of `[ai.*]` bindings without any provider-specific
code — the OpenAI API is the contract.

- `GET /` — echoes the injected binding env as JSON (`OPENAI_BASE_URL`,
  `OPENAI_MODEL`, every `NERDIT_AI_*` URL/MODEL value; key variables are
  reported as *name + set/unset only*, values are never echoed).
- `GET /chat[?binding=cheap][&prompt=...]` — forwards one chat-completion call
  to the selected binding's base URL using `fetch` (no SDK). The default
  binding is the local Ollama model; `?binding=cheap` uses the external API.

Bindings (see `nerdit.toml`):

- `[ai.default]` — `provider = "ollama"`, `model = "llama3.1:8b"` → requires a
  served model: `nerdit serve llama3.1:8b --gpus 1`.
- `[ai.cheap]` — `provider = "api"` with `api_key = "${secrets.OPENAI_KEY}"` →
  requires `nerdit secrets set ai-app OPENAI_KEY=sk-...`.

For deployment and model setup, follow the [AI services guide](https://docs.nerdit.ai/engine/ai).

The automated CI counterpart of this fixture is `tests/test_p5_northstar.py`.
