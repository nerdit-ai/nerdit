// Nerdit P5 north-star fixture app.
//
// Dependency-free Node server (built-in http + global fetch, Node 18+). It
// exercises the [ai.*] binding contract Nerdit injects at launch:
//
//   GET /      -> JSON echo of the binding env: URL/MODEL values, and for every
//                 *KEY* variable only its NAME + whether it is set (never the
//                 value — keys must not leak through an HTTP endpoint).
//   GET /chat  -> forwards ONE chat-completion call to OPENAI_BASE_URL using
//                 fetch (no SDK dependency). `?binding=<name>` switches to the
//                 NERDIT_AI_<NAME>_* triple (e.g. /chat?binding=cheap).
//
// Listens on process.env.PORT (injected by Nerdit; Heroku/Cloud-Run convention).

'use strict';

const http = require('http');

const PORT = parseInt(process.env.PORT || '3000', 10);

// Collect the injected binding env. Values are echoed for URL/MODEL vars;
// *KEY* vars are reported as { set: true/false } only.
function bindingEnv() {
  const out = {};
  const names = ['OPENAI_BASE_URL', 'OPENAI_MODEL'].concat(
    Object.keys(process.env).filter(
      (k) => k.startsWith('NERDIT_AI_') && (k.endsWith('_URL') || k.endsWith('_MODEL'))
    )
  );
  for (const name of names) {
    if (process.env[name] !== undefined) out[name] = process.env[name];
  }
  const keyNames = ['OPENAI_API_KEY'].concat(
    Object.keys(process.env).filter((k) => k.startsWith('NERDIT_AI_') && k.endsWith('_KEY'))
  );
  for (const name of keyNames) {
    out[name] = { set: process.env[name] !== undefined };
  }
  return out;
}

// Resolve the (base_url, api_key, model) triple for a binding name.
// 'default' -> the plain OPENAI_* vars; anything else -> NERDIT_AI_<NAME>_*.
function resolveBinding(name) {
  if (!name || name === 'default') {
    return {
      baseUrl: process.env.OPENAI_BASE_URL,
      apiKey: process.env.OPENAI_API_KEY,
      model: process.env.OPENAI_MODEL,
    };
  }
  const prefix = `NERDIT_AI_${name.toUpperCase()}`;
  return {
    baseUrl: process.env[`${prefix}_URL`],
    apiKey: process.env[`${prefix}_KEY`],
    model: process.env[`${prefix}_MODEL`],
  };
}

async function handleChat(url, res) {
  const bindingName = url.searchParams.get('binding') || 'default';
  const binding = resolveBinding(bindingName);
  if (!binding.baseUrl || !binding.model) {
    res.writeHead(503, { 'Content-Type': 'application/json' });
    res.end(
      JSON.stringify({
        error: `binding '${bindingName}' is not wired (no base_url/model in env)`,
      })
    );
    return;
  }
  const prompt = url.searchParams.get('prompt') || 'Say hi in exactly five words.';
  try {
    const upstream = await fetch(`${binding.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${binding.apiKey || ''}`,
      },
      body: JSON.stringify({
        model: binding.model,
        messages: [{ role: 'user', content: prompt }],
      }),
    });
    const body = await upstream.json();
    res.writeHead(upstream.ok ? 200 : 502, { 'Content-Type': 'application/json' });
    res.end(
      JSON.stringify({
        binding: bindingName,
        model: binding.model,
        upstream_status: upstream.status,
        reply: body.choices?.[0]?.message?.content ?? null,
      })
    );
  } catch (err) {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ binding: bindingName, error: String(err) }));
  }
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  if (req.method === 'GET' && url.pathname === '/') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ binding_env: bindingEnv() }, null, 2));
    return;
  }
  if (req.method === 'GET' && url.pathname === '/chat') {
    handleChat(url, res);
    return;
  }
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

server.listen(PORT, () => {
  console.log(`ai-app listening on port ${PORT}`);
});
