'use client';

import { useState, useEffect, useCallback, useRef, SetStateAction } from 'react';

function restore<T>(value: unknown, fallback: T): T {
  if (typeof fallback === 'number') return (typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : fallback) as T;
  if (fallback && typeof fallback === 'object' && !Array.isArray(fallback)) {
    const source = value && typeof value === 'object' ? value as Record<string, unknown> : {};
    return Object.fromEntries(Object.entries(fallback).map(([key, item]) => [key, restore(source[key], item)])) as T;
  }
  return (typeof value === typeof fallback ? value : fallback) as T;
}

export function usePersistedState<T>(key: string, defaultValue: T): [T, (val: SetStateAction<T>) => void] {
  const defaults = useRef(defaultValue);
  const [value, setValue] = useState(defaultValue);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(key);
      if (stored !== null) setValue(restore(JSON.parse(stored), defaults.current));
    } catch { /* Practice remains available without local storage. */ }
    setHydrated(true);
  }, [key]);

  useEffect(() => {
    if (!hydrated) return;
    try { localStorage.setItem(key, JSON.stringify(value)); }
    catch { /* Device-local streak persistence is optional. */ }
  }, [key, value, hydrated]);

  const setPersisted = useCallback((next: SetStateAction<T>) => { setValue(next); }, []);
  return [value, setPersisted];
}
