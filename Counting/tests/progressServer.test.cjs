const test = require('node:test');
const assert = require('node:assert/strict');
const { createProgressHandler, PROGRESS_COMPARE_AND_SET } = require('../lib/progressServer.ts');
const { emptyProgress, MAX_PROGRESS_BYTES } = require('../lib/progressSync.ts');
const { addQuickResult, createQuickQuestions, createQuickResult, recordQuickAnswer } = require('../lib/quickPlay.ts');

const secret = 'test-only-sync-key-with-32-characters-minimum';
const env = { COUNTING_SYNC_KEY: secret, UPSTASH_REDIS_REST_URL: 'https://redis.example.test',
  UPSTASH_REDIS_REST_TOKEN: 'test-storage-token', VERCEL_ENV: 'production' };

function request(method = 'GET', body, key = secret, headers = {}) {
  return new Request('https://trainer.example.test/api/progress', { method,
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json', ...headers },
    ...(body === undefined ? {} : { body: typeof body === 'string' ? body : JSON.stringify(body) }),
  });
}

function memoryStore(initial = null) {
  let value = initial;
  const commands = [];
  return {
    commands,
    get value() { return value; },
    fetch: async (_url, options) => {
      const command = JSON.parse(options.body);
      commands.push(command);
      if (command[0] === 'GET') return Response.json({ result: value });
      assert.equal(command[0], 'EVAL');
      assert.equal(command[1], PROGRESS_COMPARE_AND_SET);
      assert.equal(command[2], 1);
      const matches = command[4] === '0' ? value === null : value === command[5];
      if (matches) value = command[6];
      return Response.json({ result: Number(matches) });
    },
  };
}

function progressWithRun(seed, timeZone = 'America/Toronto') {
  const progress = emptyProgress(timeZone);
  const questions = createQuickQuestions(seed, 'counting');
  let answers = [];
  for (let index = 0; index < 5; index++) {
    answers = recordQuickAnswer(questions, answers, index, questions[index].target, (index + 1) * 100);
  }
  const result = createQuickResult({ seed, mode: 'counting', startedAt: `2026-09-26T12:00:${String(seed).padStart(2, '0')}.000Z`,
    endedAt: `2026-09-26T12:01:${String(seed).padStart(2, '0')}.000Z`, elapsedMs: 60_000, endReason: 'timer', answers });
  progress.quick = addQuickResult(progress.quick, result);
  return progress;
}

test('sync requires valid server configuration and a private key before reading storage', async () => {
  const store = memoryStore();
  for (const changes of [
    { COUNTING_SYNC_KEY: undefined }, { COUNTING_SYNC_KEY: 'short' },
    { UPSTASH_REDIS_REST_TOKEN: undefined }, { UPSTASH_REDIS_REST_URL: 'http://redis.example.test' },
    { COUNTING_SYNC_NAMESPACE: 'invalid prefix' }, { COUNTING_SYNC_TIME_ZONE: 'not/a/timezone' },
  ]) {
    const result = await createProgressHandler({ env: { ...env, ...changes }, fetch: store.fetch })(request());
    assert.equal(result.status, 503);
    assert.equal((await result.json()).code, 'not_configured');
    assert.equal(result.headers.get('cache-control'), 'no-store');
  }
  const handler = createProgressHandler({ env, fetch: store.fetch });
  for (const key of ['', 'incorrect-key', 'é'.repeat(40), secret + 'x']) {
    const result = await handler(request('GET', undefined, key));
    assert.equal(result.status, 401);
    assert.equal((await result.json()).code, 'unauthorized');
    assert.equal(result.headers.get('access-control-allow-origin'), null);
  }
  assert.equal(store.commands.length, 0);
});

test('GET returns an empty document in the server timezone and isolates deployment keys', async () => {
  const keys = [];
  for (const deployment of ['production', 'preview', 'development', undefined]) {
    const store = memoryStore();
    const handler = createProgressHandler({ env: { ...env, VERCEL_ENV: deployment }, fetch: store.fetch });
    const result = await handler(request());
    assert.equal(result.status, 200);
    assert.deepEqual(await result.json(), { progress: emptyProgress('America/Toronto') });
    assert.equal(store.commands.length, 1, 'a GET never writes an empty document');
    keys.push(store.commands[0][1]);
  }
  assert.notEqual(keys[0], keys[1]);
  assert.notEqual(keys[0], keys[2]);
  assert.equal(keys[2], keys[3]);
});

test('POST persists canonical merged progress and the server timezone wins on first upload', async () => {
  const store = memoryStore();
  const handler = createProgressHandler({ env, fetch: store.fetch });
  const first = await handler(request('POST', progressWithRun(1, 'Asia/Tokyo')));
  assert.equal(first.status, 200);
  const uploaded = (await first.json()).progress;
  assert.equal(uploaded.timeZone, 'America/Toronto');
  assert.equal(uploaded.quick.sessions.length, 1);
  const second = await handler(request('POST', progressWithRun(2)));
  assert.equal(second.status, 200);
  const merged = (await second.json()).progress;
  assert.equal(merged.quick.sessions.length, 2);
  assert.deepEqual(JSON.parse(store.value), merged);
  const read = await handler(request());
  assert.deepEqual((await read.json()).progress, merged);
  assert.ok(store.commands.every(command => !['EXPIRE', 'DEL', 'SET'].includes(command[0])), 'history has no TTL or unguarded SET');
  assert.equal(first.headers.get('cache-control'), 'no-store');
});

test('simultaneous device uploads retry their compare-and-set and retain both sessions', async () => {
  const store = memoryStore();
  let initialReads = 0;
  let releaseReads;
  const bothRead = new Promise(resolve => { releaseReads = resolve; });
  const fetch = async (url, options) => {
    const command = JSON.parse(options.body);
    if (command[0] === 'GET' && initialReads < 2) {
      initialReads++;
      const snapshot = await store.fetch(url, options);
      if (initialReads === 2) releaseReads();
      await bothRead;
      return snapshot;
    }
    return store.fetch(url, options);
  };
  const handler = createProgressHandler({ env, fetch });
  const results = await Promise.all([handler(request('POST', progressWithRun(3))), handler(request('POST', progressWithRun(4)))]);
  assert.deepEqual(results.map(result => result.status), [200, 200]);
  const saved = JSON.parse(store.value);
  assert.deepEqual(saved.quick.sessions.map(session => session.seed).sort(), [3, 4]);
  assert.equal(store.commands.filter(command => command[0] === 'EVAL').length, 3, 'the stale upload retries once');
});

test('invalid and oversized incoming data cannot read or change cloud history', async () => {
  const store = memoryStore();
  const handler = createProgressHandler({ env, fetch: store.fetch });
  for (const body of ['{', 'null', '{}', { ...emptyProgress(), version: 99 }]) {
    const result = await handler(request('POST', body));
    assert.equal(result.status, 400);
    assert.equal((await result.json()).code, 'invalid_progress');
  }
  const declared = await handler(request('POST', '{}', secret, { 'Content-Length': String(MAX_PROGRESS_BYTES + 1) }));
  assert.equal(declared.status, 413);
  const actual = await handler(request('POST', ' '.repeat(MAX_PROGRESS_BYTES + 1), secret, { 'Content-Length': '1' }));
  assert.equal(actual.status, 413, 'stream bytes, not an untrusted header, determine the limit');
  assert.equal(store.commands.length, 0);
});

test('malformed stored data and storage failures never silently reset or overwrite progress', async () => {
  for (const raw of ['{', '{}', 'null']) {
    const store = memoryStore(raw);
    const handler = createProgressHandler({ env, fetch: store.fetch });
    for (const req of [request(), request('POST', progressWithRun(5))]) {
      const result = await handler(req);
      assert.equal(result.status, 503);
      assert.equal((await result.json()).code, 'unavailable');
    }
    assert.equal(store.value, raw);
    assert.ok(store.commands.every(command => command[0] === 'GET'));
  }
  for (const fetch of [
    async () => { throw new Error('Secret internal details'); },
    async () => new Response('Internal credential information', { status: 500 }),
    async () => Response.json({ error: 'WRONGTYPE private data' }),
    async () => Response.json({ result: 12 }),
  ]) {
    const result = await createProgressHandler({ env, fetch })(request('POST', progressWithRun(6)));
    assert.equal(result.status, 503);
    assert.deepEqual(await result.json(), { code: 'unavailable',
      error: 'Cloud sync is temporarily unavailable. Your progress is still saved on this device.' });
  }
});

test('storage timeouts abort the upstream call, and repeated contention has a bounded retry count', async () => {
  let aborted = false;
  const waitingFetch = async (_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener('abort', () => { aborted = true; reject(new Error('aborted')); });
  });
  const timeout = await createProgressHandler({ env, fetch: waitingFetch, timeoutMs: 5 })(request());
  assert.equal(timeout.status, 503);
  assert.equal(aborted, true);

  let writes = 0;
  const contendedFetch = async (_url, options) => {
    const command = JSON.parse(options.body);
    if (command[0] === 'GET') return Response.json({ result: null });
    writes++;
    return Response.json({ result: 0 });
  };
  const contention = await createProgressHandler({ env, fetch: contendedFetch, maxRetries: 3 })(request('POST', progressWithRun(7)));
  assert.equal(contention.status, 503);
  assert.equal(writes, 3);
});

test('Vercel KV-compatible credentials use the same durable REST protocol', async () => {
  const store = memoryStore();
  const handler = createProgressHandler({ env: { COUNTING_SYNC_KEY: secret,
    KV_REST_API_URL: env.UPSTASH_REDIS_REST_URL, KV_REST_API_TOKEN: env.UPSTASH_REDIS_REST_TOKEN }, fetch: store.fetch });
  assert.equal((await handler(request())).status, 200);
  assert.equal(store.commands.length, 1);
});
