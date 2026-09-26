import { createHash, timingSafeEqual } from 'node:crypto';
import { emptyProgress, MAX_PROGRESS_BYTES, mergeProgress, normalizeProgressTimeZone, readProgress } from './progressSync';

type Environment = Record<string, string | undefined>;
interface HandlerOptions {
  env?: Environment;
  fetch?: typeof fetch;
  timeoutMs?: number;
  maxRetries?: number;
}
interface Configuration { url: string; token: string; secret: string; key: string; timeZone: string }

// A missing value is different from an existing empty string. Comparing the
// exact saved document makes concurrent offline uploads safe without WATCH.
export const PROGRESS_COMPARE_AND_SET = `local current = redis.call('GET', KEYS[1])
if (ARGV[1] == '0' and current == false) or (ARGV[1] == '1' and current == ARGV[2]) then
  redis.call('SET', KEYS[1], ARGV[3])
  return 1
end
return 0`;

class PayloadTooLarge extends Error {}
class StorageUnavailable extends Error {}

function configuration(env: Environment): Configuration | null {
  const secret = env.COUNTING_SYNC_KEY;
  const url = env.UPSTASH_REDIS_REST_URL || env.KV_REST_API_URL;
  const token = env.UPSTASH_REDIS_REST_TOKEN || env.KV_REST_API_TOKEN;
  if (!secret || secret.length < 32 || secret.length > 256 || /\s/.test(secret) || !url || !token) return null;
  const namespace = env.COUNTING_SYNC_NAMESPACE || 'counting-trainer';
  if (!/^[a-zA-Z0-9][a-zA-Z0-9:_-]{0,63}$/.test(namespace)) return null;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:' || parsed.username || parsed.password || parsed.search || parsed.hash) return null;
    const timeZone = normalizeProgressTimeZone(env.COUNTING_SYNC_TIME_ZONE || 'America/Toronto');
    // Even if preview deployments inherit production credentials, their data
    // lives under a separate key. Local development is isolated as well.
    const deployment = env.VERCEL_ENV === 'production' ? 'production'
      : env.VERCEL_ENV === 'preview' ? 'preview' : 'development';
    return { url: parsed.toString().replace(/\/$/, ''), token, secret,
      key: `${namespace}:${deployment}:progress:v1`, timeZone };
  } catch { return null; }
}

function authorized(request: Request, secret: string): boolean {
  const authorization = request.headers.get('authorization') || '';
  if (!authorization.startsWith('Bearer ') || authorization.length > 263) return false;
  // Hashes have equal byte lengths, including when a wrong key contains UTF-8.
  return timingSafeEqual(createHash('sha256').update(authorization.slice(7)).digest(),
    createHash('sha256').update(secret).digest());
}

function response(body: unknown, status = 200, extraHeaders: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: {
    'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
    Pragma: 'no-cache', 'X-Content-Type-Options': 'nosniff', Vary: 'Authorization', ...extraHeaders,
  } });
}

async function readBody(request: Request): Promise<string> {
  const declaredLength = request.headers.get('content-length');
  if (declaredLength && /^\d+$/.test(declaredLength) && Number(declaredLength) > MAX_PROGRESS_BYTES) {
    throw new PayloadTooLarge();
  }
  if (!request.body) return '';
  const reader = request.body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let size = 0;
  let text = '';
  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      size += chunk.value.byteLength;
      if (size > MAX_PROGRESS_BYTES) {
        await reader.cancel().catch(() => undefined);
        throw new PayloadTooLarge();
      }
      text += decoder.decode(chunk.value, { stream: true });
    }
    return text + decoder.decode();
  } finally { reader.releaseLock(); }
}

/** The route has no dependency on a particular host and can be tested with an
 * injected REST transport. Durable storage must be available before accepting
 * a write; there is deliberately no process-memory fallback. */
export function createProgressHandler(options: HandlerOptions = {}) {
  const requestFetch = options.fetch ?? globalThis.fetch;
  const timeoutMs = options.timeoutMs ?? 8000;
  const maxRetries = options.maxRetries ?? 5;

  return async function handleProgress(request: Request): Promise<Response> {
    const config = configuration(options.env ?? process.env);
    if (!config) return response({ code: 'not_configured', error: 'Cloud sync is not configured.' }, 503);
    if (!authorized(request, config.secret)) return response({ code: 'unauthorized', error: 'Enter the correct sync key.' }, 401);
    if (request.method !== 'GET' && request.method !== 'POST') {
      return response({ code: 'method_not_allowed', error: 'Use GET or POST.' }, 405, { Allow: 'GET, POST' });
    }

    async function command(args: Array<string | number>): Promise<unknown> {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const upstream = await requestFetch(config!.url, {
          method: 'POST', headers: { Authorization: `Bearer ${config!.token}`, 'Content-Type': 'application/json' },
          body: JSON.stringify(args), signal: controller.signal, cache: 'no-store', redirect: 'error',
        });
        if (!upstream.ok) throw new StorageUnavailable();
        const payload = await upstream.json();
        if (!payload || typeof payload !== 'object' || 'error' in payload || !('result' in payload)) throw new StorageUnavailable();
        return payload.result;
      } catch { throw new StorageUnavailable(); }
      finally { clearTimeout(timer); }
    }

    async function storedProgress() {
      const raw = await command(['GET', config!.key]);
      if (raw !== null && typeof raw !== 'string') throw new StorageUnavailable();
      try {
        return { raw, progress: raw === null ? emptyProgress(config!.timeZone) : readProgress(raw) };
      } catch { throw new StorageUnavailable(); }
    }

    let incoming;
    if (request.method === 'POST') {
      try { incoming = readProgress(await readBody(request)); }
      catch (error) {
        if (error instanceof PayloadTooLarge) return response({ code: 'payload_too_large', error: 'This progress file is too large to sync.' }, 413);
        return response({ code: 'invalid_progress', error: 'This progress file is invalid.' }, 400);
      }
    }

    try {
      if (request.method === 'GET') return response({ progress: (await storedProgress()).progress });
      for (let attempt = 0; attempt < maxRetries; attempt++) {
        const current = await storedProgress();
        const progress = mergeProgress(current.progress, incoming!);
        const serialized = JSON.stringify(progress);
        if (Buffer.byteLength(serialized, 'utf8') > MAX_PROGRESS_BYTES) throw new PayloadTooLarge();
        const result = await command(['EVAL', PROGRESS_COMPARE_AND_SET, 1, config.key,
          current.raw === null ? '0' : '1', current.raw ?? '', serialized]);
        if (result === 1) return response({ progress });
        if (result !== 0) throw new StorageUnavailable();
      }
      throw new StorageUnavailable();
    } catch (error) {
      if (error instanceof PayloadTooLarge) return response({ code: 'payload_too_large', error: 'Your combined progress is too large to sync.' }, 413);
      return response({ code: 'unavailable', error: 'Cloud sync is temporarily unavailable. Your progress is still saved on this device.' }, 503);
    }
  };
}
