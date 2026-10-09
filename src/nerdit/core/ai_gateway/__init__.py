"""The machine's AI gateway: an OpenAI-compatible proxy served inside `nerditd`.

App containers reach it on the models bridge with a per-service virtual key;
the gateway maps the request's `model` alias to a provider model and key that
never leave the machine. See `app.py` (listener), `proxy.py` (forwarding) and
`keys.py` (virtual keys).
"""
