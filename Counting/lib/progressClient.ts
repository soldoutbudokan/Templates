import { readQuickHistory } from './quickPlay';
import { readTableHistory } from './tableHistory';
import { readHistory } from './training';
import { emptyProgress, MAX_PROGRESS_BYTES, mergeProgress, ProgressData, readProgress } from './progressSync';

export const PROGRESS_STORAGE_KEY = 'counting-progress-v1';
export const SYNC_KEY_STORAGE_KEY = 'counting-sync-key-v1';
const MAX_SYNC_PASSES = 3;

export type ProgressStatus = 'local' | 'syncing' | 'synced' | 'pending' | 'error' | 'unavailable';
export interface ProgressSnapshot {
  progress: ProgressData;
  linked: boolean;
  status: ProgressStatus;
  message: string;
  lastSynced: string | null;
  localWarning: string;
}
export interface ProgressClientOptions {
  storage: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
  fetcher: typeof fetch;
  online: () => boolean;
  now?: () => Date;
  timeoutMs?: number;
}

class SyncFailure extends Error {
  constructor(readonly status: ProgressStatus, message: string) { super(message); }
}
function validKey(key: unknown): key is string {
  return typeof key === 'string' && /^\S{32,256}$/.test(key);
}
function serialized(progress: ProgressData): string { return JSON.stringify(progress); }
function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Browser persistence and transport without browser globals. The UI owns its
 * lifecycle and retries; all successful responses merge with the latest local
 * data, including results earned while a request was in flight.
 */
export class ProgressClient {
  private snapshot: ProgressSnapshot = {
    progress: emptyProgress(), linked: false, status: 'local', message: 'Saved on this device.',
    lastSynced: null, localWarning: '',
  };
  private listeners = new Set<() => void>();
  private key = '';
  private observedStoredKey: string | null = null;
  private generation = 0;
  private disposed = false;
  private initialized = false;
  private initializeTask: Promise<void> | null = null;
  private syncTask: Promise<void> | null = null;
  private controller: AbortController | null = null;
  private progressWarning = '';
  private keyWarning = '';
  private cacheDiffers = false;
  private now: () => Date;

  constructor(private options: ProgressClientOptions) { this.now = options.now ?? (() => new Date()); }

  getSnapshot = (): ProgressSnapshot => this.snapshot;
  getKey = (): string => this.key;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  private publish(patch: Partial<ProgressSnapshot> = {}) {
    if (this.disposed) return;
    const next = { ...this.snapshot, ...patch, localWarning: [this.progressWarning, this.keyWarning].filter(Boolean).join(' ') };
    if (Object.keys(next).every(key => next[key as keyof ProgressSnapshot] === this.snapshot[key as keyof ProgressSnapshot])) return;
    this.snapshot = next;
    for (const listener of Array.from(this.listeners)) listener();
  }

  private localMessage(): string {
    return this.progressWarning ? 'Progress is available in this open page.' : 'Progress is saved on this device.';
  }

  /** Abort and a deadline race both matter: some transports ignore AbortSignal. */
  private async request<T>(init: RequestInit, controller: AbortController, consume: (response: Response) => Promise<T>): Promise<T> {
    let onAbort = () => {};
    const interrupted = new Promise<T>((_, reject) => {
      onAbort = () => reject(new SyncFailure('pending', 'Sync could not finish. It will retry when a connection is available.'));
      controller.signal.addEventListener('abort', onAbort, { once: true });
      if (controller.signal.aborted) onAbort();
    });
    const timer = setTimeout(() => controller.abort(), this.options.timeoutMs ?? 20_000);
    try {
      return await Promise.race([
        this.options.fetcher('/api/progress', { ...init, signal: controller.signal }).then(consume), interrupted,
      ]);
    } finally {
      clearTimeout(timer);
      controller.signal.removeEventListener('abort', onAbort);
    }
  }

  private readStored(key: string): { readable: boolean; value: string | null } {
    try { return { readable: true, value: this.options.storage.getItem(key) }; }
    catch {
      if (key === SYNC_KEY_STORAGE_KEY) this.keyWarning = 'This browser could not read its saved sync key.';
      else this.progressWarning = 'This browser could not read saved progress. New results remain available while this page is open.';
      return { readable: false, value: null };
    }
  }

  private persistProgress(progress: ProgressData) {
    try {
      this.options.storage.setItem(PROGRESS_STORAGE_KEY, serialized(progress));
      this.progressWarning = '';
    } catch {
      this.progressWarning = 'This browser could not save progress. Keep this page open until cloud sync succeeds.';
    }
  }

  private localProgress(): ProgressData {
    this.cacheDiffers = false;
    const stored = this.readStored(PROGRESS_STORAGE_KEY);
    if (!stored.readable || stored.value === null) return this.snapshot.progress;
    try {
      const cached = readProgress(stored.value);
      const merged = mergeProgress(this.snapshot.progress, cached);
      this.cacheDiffers = serialized(merged) !== serialized(cached);
      return merged;
    }
    catch {
      this.progressWarning = 'Some saved progress could not be read. Your current results are still available.';
      return this.snapshot.progress;
    }
  }

  private invalidateRequests() {
    this.generation++;
    this.controller?.abort();
    this.controller = null;
    this.syncTask = null;
  }

  private current(generation: number): boolean { return !this.disposed && generation === this.generation; }

  initialize(): Promise<void> {
    if (this.disposed) return Promise.resolve();
    if (this.initialized) return this.initializeTask ?? Promise.resolve();
    this.initialized = true;
    const cache = this.readStored(PROGRESS_STORAGE_KEY);
    let progress = this.snapshot.progress;
    if (cache.readable && cache.value !== null) {
      try { progress = mergeProgress(readProgress(cache.value), progress); }
      catch { this.progressWarning = 'Saved progress could not be read. The saved copy has been left in place.'; }
    } else if (cache.readable) {
      // Old keys remain as backups. Only a missing new cache triggers migration;
      // an intentionally cleared history must never return from a legacy key.
      const quick = this.readStored('counting-quick-play-v1');
      const guided = this.readStored('counting-coach-history-v1');
      const table = this.readStored('counting-table-history-v1');
      let migrationWarning = '';
      if (quick.readable) {
        const restored = readQuickHistory(quick.value);
        progress = { ...progress, quick: restored.history };
        if (restored.reset) migrationWarning = 'Some older practice records could not be imported. Their original saved copies remain on this device.';
      }
      if (guided.readable) {
        try {
          const sessions = Array.from(new Map(readHistory(guided.value).map(result => [result.id, result])).values());
          progress = { ...progress, guided: readProgress(serialized({ ...emptyProgress(), guided: sessions })).guided };
        }
        catch { migrationWarning = 'Some older practice records could not be imported. Their original saved copies remain on this device.'; }
      }
      if (table.readable) {
        try {
          const sessions = Array.from(new Map(readTableHistory(table.value).map(result => [result.id, result])).values());
          progress = { ...progress, table: readProgress(serialized({ ...emptyProgress(), table: sessions })).table };
        }
        catch { migrationWarning = 'Some older practice records could not be imported. Their original saved copies remain on this device.'; }
      }
      // Older formats allow larger archives. Keep their saved backups intact,
      // but trim the oldest imported reviews to fit the shared document limit.
      while (new TextEncoder().encode(serialized(progress)).length > MAX_PROGRESS_BYTES) {
        migrationWarning = 'Some older reviews were too large to import. Their original saved copies remain on this device.';
        const serializedPart = (part: 'guided' | 'table' | 'quick') =>
          JSON.stringify(part === 'quick' ? progress.quick.sessions : progress[part]);
        const kinds: Array<'guided' | 'table' | 'quick'> = ['guided', 'table', 'quick'];
        const kind = kinds.sort((left, right) => serializedPart(right).length - serializedPart(left).length)[0];
        if (kind === 'quick') progress = { ...progress, quick: { ...progress.quick, sessions: progress.quick.sessions.slice(1) } };
        else progress = { ...progress, [kind]: progress[kind].slice(1) };
      }
      try { progress = readProgress(serialized(progress)); }
      catch {
        progress = this.snapshot.progress;
        migrationWarning = 'Older practice records could not be imported. Their original saved copies remain on this device.';
      }
      this.persistProgress(progress);
      if (migrationWarning) this.progressWarning = migrationWarning;
    }
    const remembered = this.readStored(SYNC_KEY_STORAGE_KEY);
    if (remembered.readable) this.observedStoredKey = remembered.value;
    if (remembered.readable && remembered.value !== null) {
      if (validKey(remembered.value)) this.key = remembered.value;
      else this.keyWarning = 'The saved sync key is invalid. Reconnect this device.';
    }
    this.publish({ progress, linked: Boolean(this.key), message: this.localMessage() });
    this.initializeTask = this.key ? this.sync() : this.probeConfiguration();
    return this.initializeTask;
  }

  private async probeConfiguration(): Promise<void> {
    if (!this.options.online()) return;
    const generation = this.generation;
    const controller = new AbortController();
    this.controller = controller;
    try {
      const response = await this.request({
        method: 'GET', cache: 'no-store', credentials: 'omit',
        headers: { Accept: 'application/json' },
      }, controller, async response => ({ status: response.status, body: response.status === 503 ? await response.json() as unknown : null }));
      if (!this.current(generation)) return;
      if (response.status === 503) {
        const body = response.body;
        if (this.current(generation) && isObject(body) && body.code === 'not_configured') {
          this.publish({ status: 'unavailable', message: `Cloud sync has not been configured yet. ${this.localMessage()}` });
        }
      }
      // A configured server returns 401 until the private key is supplied.
    } catch { /* Local practice needs neither a network connection nor a key. */ }
    finally { if (this.current(generation) && this.controller === controller) this.controller = null; }
  }

  private async responseProgress(response: Response): Promise<ProgressData> {
    let body: unknown;
    try { body = await response.json(); }
    catch { throw new SyncFailure('error', 'Cloud progress could not be read. Your local progress is safe.'); }
    if (!response.ok) {
      if (response.status === 401) throw new SyncFailure('error', 'The sync key was not accepted. Check the key and reconnect.');
      if (isObject(body) && body.code === 'not_configured') {
        throw new SyncFailure('unavailable', 'Cloud sync has not been configured yet. Progress is saved on this device.');
      }
      throw new SyncFailure('pending', 'Cloud sync is temporarily unavailable. Your progress is saved here and will retry.');
    }
    try {
      if (!isObject(body) || !isObject(body.progress)) throw new Error('Missing progress.');
      return readProgress(JSON.stringify(body.progress));
    } catch { throw new SyncFailure('error', 'Cloud progress could not be read. Your local progress is safe.'); }
  }

  private reportFailure(error: unknown, generation: number) {
    if (!this.current(generation)) return;
    if (error instanceof SyncFailure) this.publish({ status: error.status,
      message: this.progressWarning ? `${error.message.split('. ')[0]}. ${this.localMessage()}` : error.message });
    else this.publish({ status: 'pending', message: this.options.online()
      ? `Cloud sync is temporarily unavailable. ${this.localMessage()}`
      : `Offline. ${this.localMessage()} Sync will retry when you are online.` });
  }

  async connect(key: string): Promise<boolean> {
    if (this.disposed) return false;
    if (!validKey(key)) {
      this.publish({ status: 'error', message: 'Enter the complete sync key, with no spaces.' });
      return false;
    }
    if (!this.options.online()) {
      this.publish({ status: 'pending', message: `Connect when you are online. ${this.localMessage()}` });
      return false;
    }
    this.invalidateRequests();
    const generation = this.generation;
    const controller = new AbortController();
    this.controller = controller;
    this.publish({ status: 'syncing', message: 'Connecting this device…' });
    try {
      const remote = await this.request({
        method: 'GET', cache: 'no-store', credentials: 'omit',
        headers: { Accept: 'application/json', Authorization: `Bearer ${key}` },
      }, controller, response => this.responseProgress(response));
      if (!this.current(generation)) return false;
      const progress = mergeProgress(remote, this.localProgress());
      this.key = key;
      try {
        this.options.storage.setItem(SYNC_KEY_STORAGE_KEY, key);
        this.observedStoredKey = key;
        this.keyWarning = '';
      } catch { this.keyWarning = 'This browser could not remember the sync key. Reconnect after closing this page.'; }
      this.persistProgress(progress);
      this.publish({ progress, linked: true });
      if (this.controller === controller) this.controller = null;
      await this.sync();
      return this.current(generation) && this.key === key;
    } catch (error) {
      this.reportFailure(error, generation);
      return false;
    } finally { if (this.current(generation) && this.controller === controller) this.controller = null; }
  }

  sync(): Promise<void> {
    if (this.disposed || !this.key) return Promise.resolve();
    if (this.syncTask) return this.syncTask;
    if (!this.options.online()) {
      this.publish({ status: 'pending', message: `Offline. ${this.localMessage()} Sync will retry when you are online.` });
      return Promise.resolve();
    }
    const generation = this.generation;
    const key = this.key;
    const controller = new AbortController();
    this.controller = controller;
    const task = Promise.resolve().then(() => this.upload(generation, key, controller)).finally(() => {
      if (this.current(generation) && this.syncTask === task) {
        this.syncTask = null;
        if (this.controller === controller) this.controller = null;
      }
    });
    this.syncTask = task;
    return task;
  }

  private async upload(generation: number, key: string, controller: AbortController): Promise<void> {
    this.publish({ status: 'syncing', message: 'Syncing progress…' });
    try {
      for (let pass = 0; pass < MAX_SYNC_PASSES; pass++) {
        if (!this.current(generation)) return;
        const local = this.localProgress();
        if (serialized(local) !== serialized(this.snapshot.progress)) this.publish({ progress: local });
        const remote = await this.request({
          method: 'POST', cache: 'no-store', credentials: 'omit',
          headers: { Accept: 'application/json', 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
          body: serialized(this.snapshot.progress),
        }, controller, response => this.responseProgress(response));
        if (!this.current(generation)) return;
        const progress = mergeProgress(remote, this.localProgress());
        this.persistProgress(progress);
        this.publish({ progress, lastSynced: this.now().toISOString() });
        if (serialized(this.snapshot.progress) === serialized(remote)) {
          this.publish({ status: 'synced', message: 'Progress is synced across your devices.' });
          return;
        }
        // A result arrived during the request. Upload its union with the remote
        // response, rather than letting that response replace the new result.
      }
      this.publish({ status: 'pending', message: `${this.localMessage()} Cloud sync will retry shortly.` });
    } catch (error) { this.reportFailure(error, generation); }
  }

  update(updater: (progress: ProgressData) => ProgressData): void {
    if (this.disposed) return;
    const progress = readProgress(serialized(updater(readProgress(serialized(this.localProgress())))));
    this.persistProgress(progress);
    this.publish({ progress, ...(this.key ? {} : { status: this.snapshot.status === 'unavailable' ? 'unavailable' as const : 'local' as const,
      message: this.snapshot.status === 'unavailable' ? `Cloud sync has not been configured yet. ${this.localMessage()}` : this.localMessage() }) });
    void this.sync();
  }

  refreshFromStorage(externalProgress?: string | null): void {
    if (this.disposed) return;
    let progress = this.localProgress();
    // The event carries a write which may already have been overwritten in
    // storage. Reading only the current key can lose a concurrently saved run.
    if (typeof externalProgress === 'string') {
      try { progress = mergeProgress(progress, readProgress(externalProgress)); }
      catch { this.progressWarning = 'A progress update from another tab could not be read. Your current results are still available.'; }
    }
    const changed = serialized(progress) !== serialized(this.snapshot.progress);
    if (changed || this.cacheDiffers) {
      this.persistProgress(progress);
      this.publish({ progress });
    } else this.publish();
    const remembered = this.readStored(SYNC_KEY_STORAGE_KEY);
    if (!remembered.readable) { this.publish(); return; }
    const externalKey = remembered.value ?? '';
    const externallyChanged = remembered.value !== this.observedStoredKey;
    this.observedStoredKey = remembered.value;
    if (externalKey !== this.key && externallyChanged) {
      this.invalidateRequests();
      this.key = '';
      this.publish({ linked: false, status: 'local', message: this.localMessage(), lastSynced: null });
      if (validKey(externalKey)) void this.connect(externalKey);
      else if (externalKey) {
        this.keyWarning = 'The saved sync key is invalid. Reconnect this device.';
        this.publish();
      }
    } else if (changed) void this.sync();
  }

  disconnect(): void {
    if (this.disposed) return;
    this.invalidateRequests();
    this.key = '';
    try { this.options.storage.removeItem(SYNC_KEY_STORAGE_KEY); this.observedStoredKey = null; this.keyWarning = ''; }
    catch { this.keyWarning = 'This browser could not remove its saved sync key. Clear this site’s saved data to disconnect it after closing this page.'; }
    this.publish({ linked: false, status: 'local', message: `Disconnected. ${this.localMessage()}`, lastSynced: null });
  }

  dispose(): void {
    this.disposed = true;
    this.invalidateRequests();
    this.listeners.clear();
  }
}
