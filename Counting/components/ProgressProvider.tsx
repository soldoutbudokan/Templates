'use client';

import { createContext, useCallback, useContext, useEffect, useRef, useState, ReactNode } from 'react';
import { ProgressClient, ProgressSnapshot, PROGRESS_STORAGE_KEY, SYNC_KEY_STORAGE_KEY } from '@/lib/progressClient';
import { clearProgressHistory, emptyProgress, mergeProgress, progressDateKey, ProgressData } from '@/lib/progressSync';
import { addQuickResult, QuickSessionResult } from '@/lib/quickPlay';
import { SessionResult } from '@/lib/training';
import { TableSessionResult } from '@/lib/blackjack';

interface ProgressContextValue extends ProgressSnapshot {
  loaded: boolean;
  today: string;
  connect: (key: string) => Promise<boolean>;
  disconnect: () => void;
  sync: () => Promise<void>;
  getKey: () => string;
  saveQuick: (result: QuickSessionResult) => void;
  saveGuided: (result: SessionResult) => void;
  saveTable: (result: TableSessionResult) => void;
  clearHistory: (kind: 'guided' | 'table') => void;
  importProgress: (data: ProgressData) => void;
}

const ProgressContext = createContext<ProgressContextValue | null>(null);
const initialSnapshot: ProgressSnapshot = {
  progress: emptyProgress(), linked: false, status: 'local', message: '', lastSynced: null, localWarning: '',
};

export function ProgressProvider({ children }: { children: ReactNode }) {
  const client = useRef<ProgressClient | null>(null);
  const [snapshot, setSnapshot] = useState(initialSnapshot);
  const [loaded, setLoaded] = useState(false);
  const [clock, setClock] = useState<Date | null>(null);

  useEffect(() => {
    // Access is deferred so blocked browser storage cannot prevent rendering.
    const storage = {
      getItem: (key: string) => window.localStorage.getItem(key),
      setItem: (key: string, value: string) => window.localStorage.setItem(key, value),
      removeItem: (key: string) => window.localStorage.removeItem(key),
    };
    const instance = new ProgressClient({ storage, fetcher: (...args) => fetch(...args), online: () => navigator.onLine });
    client.current = instance;
    let live = true;
    const unsubscribe = instance.subscribe(() => { if (live) setSnapshot(instance.getSnapshot()); });
    // initialize restores the local snapshot synchronously; network work never blocks play.
    void instance.initialize();
    setSnapshot(instance.getSnapshot());
    setLoaded(true);
    setClock(new Date());

    const refresh = () => { setClock(new Date()); void instance.sync(); };
    const storageChanged = (event: StorageEvent) => {
      if (event.key === PROGRESS_STORAGE_KEY) instance.refreshFromStorage(event.newValue);
      else if (event.key === SYNC_KEY_STORAGE_KEY || event.key === null) instance.refreshFromStorage();
    };
    const visible = () => { if (!document.hidden) refresh(); };
    const retry = window.setInterval(() => {
      setClock(new Date());
      if (!document.hidden && ['pending', 'error'].includes(instance.getSnapshot().status)) void instance.sync();
    }, 60_000);
    window.addEventListener('online', refresh);
    window.addEventListener('focus', refresh);
    window.addEventListener('storage', storageChanged);
    document.addEventListener('visibilitychange', visible);
    return () => {
      live = false; unsubscribe(); instance.dispose(); client.current = null;
      window.clearInterval(retry);
      window.removeEventListener('online', refresh);
      window.removeEventListener('focus', refresh);
      window.removeEventListener('storage', storageChanged);
      document.removeEventListener('visibilitychange', visible);
    };
  }, []);

  const connect = useCallback((key: string) => client.current?.connect(key) ?? Promise.resolve(false), []);
  const disconnect = useCallback(() => client.current?.disconnect(), []);
  const sync = useCallback(() => client.current?.sync() ?? Promise.resolve(), []);
  const getKey = useCallback(() => client.current?.getKey() ?? '', []);
  const saveQuick = useCallback((result: QuickSessionResult) => client.current?.update(data => ({
    ...data, quick: addQuickResult(data.quick, { ...result, localDate: progressDateKey(result.endedAt, data.timeZone) }),
  })), []);
  const saveGuided = useCallback((result: SessionResult) => client.current?.update(data => mergeProgress(data, {
    ...emptyProgress(data.timeZone), guided: [result],
  })), []);
  const saveTable = useCallback((result: TableSessionResult) => client.current?.update(data => mergeProgress(data, {
    ...emptyProgress(data.timeZone), table: [result],
  })), []);
  const clearHistory = useCallback((kind: 'guided' | 'table') => client.current?.update(data => clearProgressHistory(data, kind)), []);
  const importProgress = useCallback((data: ProgressData) => client.current?.update(current => mergeProgress(current, data)), []);

  return <ProgressContext.Provider value={{ ...snapshot, loaded,
    today: clock ? progressDateKey(clock, snapshot.progress.timeZone) : '',
    connect, disconnect, sync, getKey, saveQuick, saveGuided, saveTable, clearHistory, importProgress,
  }}>{children}</ProgressContext.Provider>;
}

export function useProgress() {
  const value = useContext(ProgressContext);
  if (!value) throw new Error('ProgressProvider is required.');
  return value;
}
