const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

test('chat errors do not expose malformed upstream response bodies', async () => {
  let handleRequest;
  vm.runInNewContext(readFileSync(join(__dirname, '../examples/ai-app/server.js'), 'utf8'), {
    require: (name) => {
      assert.equal(name, 'http');
      return {
        createServer: (handler) => {
          handleRequest = handler;
          return { listen() {} };
        },
      };
    },
    process: { env: { OPENAI_BASE_URL: 'https://upstream.invalid', OPENAI_MODEL: 'test' } },
    URL,
    fetch: async () => new Response('CANARY42 private upstream detail'),
  });

  let status;
  const body = await new Promise((resolve) => {
    handleRequest(
      { method: 'GET', url: '/chat', headers: { host: 'localhost' } },
      { writeHead: (code) => { status = code; }, end: resolve },
    );
  });
  assert.equal(status, 502);
  assert.equal(body.includes('CANARY42'), false);
  assert.deepEqual(JSON.parse(body), { binding: 'default', error: 'Upstream request failed' });
});
