'use client';

import { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import { Button } from './button';

const STORAGE_KEY = 'synaptic-theme';

export type ThemeMode = 'light' | 'dark' | 'system';

export function applyTheme(mode: ThemeMode) {
  const dark =
    mode === 'system'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
      : mode === 'dark';
  document.documentElement.classList.toggle('dark', dark);
}

export function readThemeMode(): ThemeMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // Storage unavailable — fall through to system default
  }
  return 'system';
}

export function writeThemeMode(mode: ThemeMode) {
  try {
    if (mode === 'system') {
      localStorage.removeItem(STORAGE_KEY);
    } else {
      localStorage.setItem(STORAGE_KEY, mode);
    }
  } catch {
    // Ignore — theme still applies for this session
  }
}

export function ThemeToggle() {
  const [isDark, setIsDark] = useState<boolean | null>(null);

  useEffect(() => {
    // The inline script in layout.tsx applies the theme class before hydration;
    // this syncs the icon to that externally-owned state.
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional mount-time read
    setIsDark(document.documentElement.classList.contains('dark'));
  }, []);

  function toggle() {
    const mode: ThemeMode = document.documentElement.classList.contains('dark')
      ? 'light'
      : 'dark';
    applyTheme(mode);
    writeThemeMode(mode);
    setIsDark(mode === 'dark');
  }

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      onClick={toggle}
      aria-label={
        isDark === null
          ? 'Toggle theme'
          : isDark
            ? 'Switch to light mode'
            : 'Switch to dark mode'
      }
    >
      {isDark === null ? (
        <span className="size-4" />
      ) : isDark ? (
        <Sun className="size-4" aria-hidden="true" />
      ) : (
        <Moon className="size-4" aria-hidden="true" />
      )}
    </Button>
  );
}
