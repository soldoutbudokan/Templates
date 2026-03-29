'use client';

import { useState, useEffect, useCallback, SetStateAction } from 'react';

export function usePersistedState<T>(key: string, defaultValue: T): [T, (val: SetStateAction<T>) => void] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === 'undefined') return defaultValue;
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? JSON.parse(stored) : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  const setPersisted = useCallback((newValue: SetStateAction<T>) => {
    setValue(prev => {
      const resolved = typeof newValue === 'function'
        ? (newValue as (prev: T) => T)(prev)
        : newValue;
      try {
        localStorage.setItem(key, JSON.stringify(resolved));
      } catch {
        // localStorage unavailable
      }
      return resolved;
    });
  }, [key]);

  // Sync with localStorage on mount (handles SSR hydration mismatch)
  useEffect(() => {
    try {
      const stored = localStorage.getItem(key);
      if (stored !== null) {
        setValue(JSON.parse(stored));
      }
    } catch {
      // localStorage unavailable
    }
  }, [key]);

  return [value, setPersisted];
}
