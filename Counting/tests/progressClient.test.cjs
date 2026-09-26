const test = require('node:test');
const assert = require('node:assert/strict');
const { ProgressClient, PROGRESS_STORAGE_KEY, SYNC_KEY_STORAGE_KEY } = require('../lib/progressClient.ts');
const { emptyProgress, mergeProgress, readProgress, clearProgressHistory } = require('../lib/progressSync.ts');
const { addQuickResult, createQuickQuestions, createQuickResult, recordQuickAnswer } = require('../lib/quickPlay.ts');
const { createTrainingPlan } = require('../lib/training.ts');
const { createTableSession, finishTableSession } = require('../lib/blackjack.ts');

const KEY = 'private-counting-key-'.repeat(3);
const NOW = '2026-09-26T18:00:00.000Z';
const tick = () => new Promise(resolve => setImmediate(resolve));
const response = (body, status = 200) => new Response(JSON.stringify(body), { status });
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};
function storage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    values, failRead: false, failWrite: false,
    getItem(key) { if (this.failRead) throw new Error('blocked'); return values.get(key) ?? null; },
    setItem(key, value) { if (this.failWrite) throw new Error('quota'); values.set(key, value); },
    removeItem(key) { if (this.failWrite) throw new Error('blocked'); values.delete(key); },
  };
}
function result(seed = 1, day = '2026-09-26') {
  const questions = createQuickQuestions(seed, 'counting');
  let answers = [];
  for (let i = 0; i < 5; i++) answers = recordQuickAnswer(questions, answers, i, questions[i].target, (i + 1) * 100);
  return createQuickResult({ seed, mode: 'counting', startedAt: `${day}T12:00:00.000Z`, endedAt: `${day}T12:01:00.000Z`,
    elapsedMs: 60000, endReason: 'timer', answers });
}
function add(client, seed, day) {
  client.update(progress => ({ ...progress, quick: addQuickResult(progress.quick, result(seed, day)) }));
}
function backend(initial = emptyProgress()) {
  const server = {
    progress: initial, calls: [], nextPost: null, postHook: null, failure: null,
    async fetch(url, options) {
      assert.equal(url, '/api/progress');
      assert.equal(options.cache, 'no-store');
      assert.equal(options.credentials, 'omit');
      this.calls.push({ method: options.method, headers: options.headers, body: options.body });
      if (this.failure) return this.failure(url, options);
      if (options.headers.Authorization !== `Bearer ${KEY}`) return response({ code: 'unauthorized' }, 401);
      if (options.method === 'GET') return response({ progress: this.progress });
      const incoming = readProgress(options.body);
      if (this.nextPost) {
        const hold = this.nextPost;
        this.nextPost = null;
        await hold.promise;
      }
      this.progress = mergeProgress(this.progress, incoming);
      const reply = response({ progress: this.progress });
      if (this.postHook) this.postHook();
      return reply;
    },
  };
  server.fetch = server.fetch.bind(server);
  return server;
}
function clientFor(saved = storage(), server = backend(), connectivity = { online: true }) {
  return new ProgressClient({ storage: saved, fetcher: server.fetch, online: () => connectivity.online, now: () => new Date(NOW) });
}
const seeds = progress => progress.quick.sessions.map(session => session.seed).sort((a, b) => a - b);

test('first load migrates all three legacy histories once and leaves their backups intact', async () => {
  const quick = addQuickResult(emptyProgress().quick, result(1, '2026-09-25'));
  const plan = createTrainingPlan(2, 'practice', 'running', 900);
  const guided = { ...plan, endedAt: NOW, completed: false, interrupted: false, assisted: false, answers: [] };
  const table = { ...finishTableSession(createTableSession({ mode: 'practice', roundLimit: 10, otherPlayers: 0, speedMs: 900 }, 3)), endedAt: NOW };
  const saved = storage({
    'counting-quick-play-v1': JSON.stringify(quick),
    'counting-coach-history-v1': JSON.stringify({ version: 1, sessions: [guided] }),
    'counting-table-history-v1': JSON.stringify({ version: 1, sessions: [table] }),
  });
  const client = clientFor(saved);
  await client.initialize();
  assert.deepEqual(seeds(client.getSnapshot().progress), [1]);
  assert.deepEqual(client.getSnapshot().progress.quick.activityDays, ['2026-09-25']);
  assert.equal(client.getSnapshot().progress.guided[0].id, guided.id);
  assert.equal(client.getSnapshot().progress.table[0].id, table.id);
  assert.ok(saved.getItem('counting-quick-play-v1'));
  assert.ok(saved.getItem(PROGRESS_STORAGE_KEY));
  client.dispose();
  saved.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(emptyProgress()));
  const second = clientFor(saved);
  await second.initialize();
  assert.deepEqual(seeds(second.getSnapshot().progress), [], 'a new empty cache must not resurrect legacy history');
  assert.deepEqual(second.getSnapshot().progress.guided, []);
  second.dispose();
});

test('only missing server configuration changes an unlinked device to unavailable', async () => {
  for (const [status, code, expected] of [[401, 'unauthorized', 'local'], [503, 'unavailable', 'local'], [503, 'not_configured', 'unavailable']]) {
    const server = backend();
    server.failure = async () => response({ code }, status);
    const client = clientFor(storage(), server);
    await client.initialize();
    assert.equal(client.getSnapshot().status, expected);
    assert.equal(client.getSnapshot().linked, false);
    client.dispose();
  }
});

test('connection authenticates before saving a key, imports local progress and uses the cloud timezone', async () => {
  const saved = storage();
  const server = backend(emptyProgress('America/Vancouver'));
  const client = clientFor(saved, server);
  await client.initialize();
  add(client, 1);
  assert.equal(await client.connect('wrong-key-'.repeat(5)), false);
  assert.equal(saved.getItem(SYNC_KEY_STORAGE_KEY), null);
  assert.equal(server.calls.filter(call => call.method === 'POST').length, 0);
  assert.deepEqual(seeds(client.getSnapshot().progress), [1]);
  assert.equal(await client.connect(KEY), true);
  assert.equal(saved.getItem(SYNC_KEY_STORAGE_KEY), KEY);
  assert.equal(client.getKey(), KEY);
  assert.deepEqual(seeds(server.progress), [1]);
  assert.equal(client.getSnapshot().progress.timeZone, 'America/Vancouver');
  assert.equal(client.getSnapshot().lastSynced, NOW);
  assert.equal(client.getSnapshot().status, 'synced');
  client.dispose();
});

test('offline practice persists immediately, then combines both devices without duplicate streak credit', async () => {
  const server = backend();
  const saved = storage();
  const connection = { online: true };
  const first = clientFor(saved, server, connection);
  const second = clientFor(storage(), server);
  await first.initialize();
  await first.connect(KEY);
  await second.initialize();
  await second.connect(KEY);
  connection.online = false;
  add(first, 1, '2026-09-25');
  assert.equal(first.getSnapshot().status, 'pending');
  assert.deepEqual(seeds(readProgress(saved.getItem(PROGRESS_STORAGE_KEY))), [1]);
  add(second, 2, '2026-09-26');
  await second.sync();
  connection.online = true;
  await first.sync();
  await second.sync();
  assert.deepEqual(seeds(first.getSnapshot().progress), [1, 2]);
  assert.deepEqual(first.getSnapshot().progress, second.getSnapshot().progress);
  await first.sync();
  assert.deepEqual(server.progress.quick.activityDays, ['2026-09-25', '2026-09-26']);
  assert.equal(server.progress.quick.sessions.length, 2);
  first.dispose(); second.dispose();
});

test('a result earned during an upload is retained and uploaded by the same sync operation', async () => {
  const saved = storage();
  const server = backend();
  const client = clientFor(saved, server);
  await client.initialize();
  await client.connect(KEY);
  const hold = deferred();
  server.nextPost = hold;
  add(client, 1);
  await tick();
  const inFlight = client.sync();
  add(client, 2);
  assert.deepEqual(seeds(readProgress(saved.getItem(PROGRESS_STORAGE_KEY))), [1, 2]);
  hold.resolve();
  await inFlight;
  assert.deepEqual(seeds(server.progress), [1, 2]);
  assert.deepEqual(seeds(client.getSnapshot().progress), [1, 2]);
  assert.equal(client.getSnapshot().status, 'synced');
  assert.equal(server.calls.filter(call => call.method === 'POST').length, 3, 'initial link, first result, then the in-flight addition');
  client.dispose();
});

test('retries are bounded if new results keep arriving during uploads', async () => {
  const server = backend();
  const client = clientFor(storage(), server);
  await client.initialize();
  await client.connect(KEY);
  let seed = 1;
  server.postHook = () => add(client, ++seed);
  add(client, seed);
  await client.sync();
  assert.equal(server.calls.filter(call => call.method === 'POST').length, 4);
  assert.equal(client.getSnapshot().status, 'pending');
  assert.deepEqual(seeds(client.getSnapshot().progress), [1, 2, 3, 4]);
  server.postHook = null;
  await client.sync();
  assert.equal(client.getSnapshot().status, 'synced');
  assert.deepEqual(seeds(server.progress), [1, 2, 3, 4]);
  client.dispose();
});

test('disconnect retains local progress and ignores an older response even if transport cannot abort', async () => {
  const saved = storage();
  const server = backend();
  const client = clientFor(saved, server);
  await client.initialize();
  await client.connect(KEY);
  const hold = deferred();
  server.nextPost = hold;
  add(client, 1);
  await tick();
  const upload = client.sync();
  client.disconnect();
  assert.equal(saved.getItem(SYNC_KEY_STORAGE_KEY), null);
  const snapshot = client.getSnapshot();
  hold.resolve();
  await upload;
  assert.equal(client.getSnapshot(), snapshot);
  assert.equal(snapshot.linked, false);
  assert.equal(snapshot.status, 'local');
  assert.deepEqual(seeds(snapshot.progress), [1]);
  client.dispose();
});

test('disconnect during key validation never remembers or adopts the late connection', async () => {
  const saved = storage();
  const server = backend();
  const client = clientFor(saved, server);
  await client.initialize();
  const hold = deferred();
  server.failure = async () => { await hold.promise; return response({ progress: emptyProgress() }); };
  const connecting = client.connect(KEY);
  client.disconnect();
  hold.resolve();
  assert.equal(await connecting, false);
  assert.equal(saved.getItem(SYNC_KEY_STORAGE_KEY), null);
  assert.equal(client.getKey(), '');
  assert.equal(client.getSnapshot().linked, false);
  client.dispose();
});

test('malformed cloud data never replaces valid local data or authenticates a new key', async () => {
  const saved = storage();
  const server = backend();
  const client = clientFor(saved, server);
  await client.initialize();
  add(client, 1);
  server.failure = async () => response({ progress: { version: 1, quick: { sessions: [] } } });
  assert.equal(await client.connect(KEY), false);
  assert.equal(client.getSnapshot().linked, false);
  assert.equal(saved.getItem(SYNC_KEY_STORAGE_KEY), null);
  assert.deepEqual(seeds(client.getSnapshot().progress), [1]);
  server.failure = null;
  await client.connect(KEY);
  const valid = saved.getItem(PROGRESS_STORAGE_KEY);
  for (const body of [{}, { progress: null }, { progress: [] }, { progress: { ...emptyProgress(), quick: { ...emptyProgress().quick, activityDays: ['not-a-date'] } } }]) {
    server.failure = async () => response(body);
    await client.sync();
    assert.equal(client.getSnapshot().status, 'error');
    assert.equal(saved.getItem(PROGRESS_STORAGE_KEY), valid);
    assert.deepEqual(seeds(client.getSnapshot().progress), [1]);
  }
  client.dispose();
});

test('blocked storage keeps progress in memory and cloud sync can still save it', async () => {
  const saved = storage();
  const server = backend();
  const client = clientFor(saved, server);
  await client.initialize();
  saved.failWrite = true;
  add(client, 1);
  assert.deepEqual(seeds(client.getSnapshot().progress), [1]);
  assert.match(client.getSnapshot().localWarning, /could not save progress/);
  assert.equal(await client.connect(KEY), true);
  assert.deepEqual(seeds(server.progress), [1]);
  assert.match(client.getSnapshot().localWarning, /could not remember the sync key/);
  client.refreshFromStorage();
  assert.equal(client.getSnapshot().linked, true, 'a key which could not be persisted stays connected in this open page');
  saved.failRead = true;
  add(client, 2);
  await client.sync();
  assert.deepEqual(seeds(client.getSnapshot().progress), [1, 2]);
  assert.deepEqual(seeds(server.progress), [1, 2]);
  saved.failRead = false; saved.failWrite = false;
  add(client, 3);
  await client.sync();
  assert.deepEqual(seeds(readProgress(saved.getItem(PROGRESS_STORAGE_KEY))), [1, 2, 3]);
  client.dispose();
});

test('queued storage event data preserves concurrent writes after one tab closes, without undoing clears', async () => {
  const saved = storage();
  const connectivity = { online: false };
  const first = clientFor(saved, backend(), connectivity);
  const second = clientFor(saved, backend(), connectivity);
  await first.initialize(); await second.initialize();
  const oldCache = saved.getItem(PROGRESS_STORAGE_KEY);
  const plan = createTrainingPlan(5, 'practice', 'running', 900);
  const guided = { ...plan, endedAt: '2026-09-25T12:00:00.000Z', completed: false, interrupted: false, assisted: false, answers: [] };
  first.update(progress => ({ ...progress, quick: addQuickResult(progress.quick, result(1)), guided: [guided] }));
  const eventFromFirstTab = saved.getItem(PROGRESS_STORAGE_KEY);
  // Both tabs read the old cache before either write becomes visible to them.
  const normalGet = saved.getItem.bind(saved);
  let supplyStaleCache = true;
  saved.getItem = key => {
    if (key === PROGRESS_STORAGE_KEY && supplyStaleCache) { supplyStaleCache = false; return oldCache; }
    return normalGet(key);
  };
  second.update(progress => clearProgressHistory({ ...progress, quick: addQuickResult(progress.quick, result(2)) }, 'guided', NOW));
  assert.deepEqual(seeds(readProgress(saved.getItem(PROGRESS_STORAGE_KEY))), [2]);
  first.dispose();
  second.refreshFromStorage(eventFromFirstTab);
  assert.deepEqual(seeds(second.getSnapshot().progress), [1, 2]);
  assert.deepEqual(second.getSnapshot().progress.guided, []);
  assert.equal(second.getSnapshot().progress.clearedBefore.guided, NOW);
  assert.deepEqual(seeds(readProgress(saved.getItem(PROGRESS_STORAGE_KEY))), [1, 2]);
  second.dispose();
});

test('a deadline bounds configuration probes, connection bodies and uploads even when fetch ignores abort', async () => {
  const saved = storage();
  const server = backend();
  const client = new ProgressClient({ storage: saved, fetcher: server.fetch, online: () => true, timeoutMs: 8 });
  server.failure = () => new Promise(() => {});
  await client.initialize();
  assert.equal(client.getSnapshot().status, 'local', 'a stalled probe does not block local practice');
  server.failure = async () => ({ status: 200, ok: true, json: () => new Promise(() => {}) });
  assert.equal(await client.connect(KEY), false);
  assert.equal(client.getSnapshot().status, 'pending');
  assert.equal(saved.getItem(SYNC_KEY_STORAGE_KEY), null);
  server.failure = null;
  await client.connect(KEY);
  server.failure = () => new Promise(() => {});
  add(client, 1);
  await client.sync();
  assert.equal(client.getSnapshot().status, 'pending');
  assert.deepEqual(seeds(client.getSnapshot().progress), [1]);
  server.failure = null;
  await client.sync();
  assert.equal(client.getSnapshot().status, 'synced');
  assert.deepEqual(seeds(server.progress), [1]);
  client.dispose();
});

test('unreadable legacy data cannot reject initialization, and blocked writes never claim durable local storage', async () => {
  const saved = storage({
    'counting-quick-play-v1': 'x'.repeat(1_000_001),
    'counting-coach-history-v1': '{',
    'counting-table-history-v1': 'null',
  });
  const connectivity = { online: false };
  const client = clientFor(saved, backend(), connectivity);
  await assert.doesNotReject(client.initialize());
  assert.match(client.getSnapshot().localWarning, /could not be imported/);
  assert.equal(saved.getItem('counting-coach-history-v1'), '{');
  saved.failWrite = true;
  add(client, 1);
  assert.match(client.getSnapshot().message, /open page/);
  assert.doesNotMatch(client.getSnapshot().message, /saved (on|here)/);
  assert.equal(await client.connect(KEY), false);
  assert.match(client.getSnapshot().message, /open page/);
  client.dispose();
});

test('cross-tab refresh unions overwritten local caches and respects external disconnects', async () => {
  const saved = storage();
  const server = backend();
  const connection = { online: false };
  const first = clientFor(saved, server, connection);
  const second = clientFor(saved, server, connection);
  await first.initialize(); await second.initialize();
  add(first, 1);
  second.refreshFromStorage();
  add(second, 2);
  first.refreshFromStorage();
  assert.deepEqual(seeds(first.getSnapshot().progress), [1, 2]);
  saved.setItem(PROGRESS_STORAGE_KEY, JSON.stringify({ ...emptyProgress(), quick: addQuickResult(emptyProgress().quick, result(1)) }));
  first.refreshFromStorage();
  assert.deepEqual(seeds(readProgress(saved.getItem(PROGRESS_STORAGE_KEY))), [1, 2], 'repair a stale cache even if the in-memory union has not changed');
  connection.online = true;
  await first.connect(KEY);
  second.refreshFromStorage();
  await tick();
  await second.sync();
  assert.equal(second.getSnapshot().linked, true);
  first.disconnect();
  second.refreshFromStorage();
  assert.equal(second.getSnapshot().linked, false);
  assert.equal(second.getKey(), '');
  assert.deepEqual(seeds(second.getSnapshot().progress), [1, 2]);
  first.dispose(); second.dispose();
});

test('snapshot is stable between changes, and corrupt caches are left intact until a new result is saved', async () => {
  const saved = storage({ [PROGRESS_STORAGE_KEY]: '{broken' });
  const client = clientFor(saved);
  await client.initialize();
  assert.equal(saved.getItem(PROGRESS_STORAGE_KEY), '{broken');
  assert.ok(client.getSnapshot().localWarning);
  const original = client.getSnapshot();
  assert.equal(client.getSnapshot(), original);
  let observed;
  const unsubscribe = client.subscribe(() => { observed = readProgress(saved.getItem(PROGRESS_STORAGE_KEY)); });
  add(client, 1);
  assert.deepEqual(seeds(observed), [1], 'persistence completes before notifying subscribers');
  assert.notEqual(client.getSnapshot(), original);
  unsubscribe();
  client.dispose();
});
