'use client';

import { useEffect, useState } from 'react';
import { AppShell } from '@/components/ui/app-shell';
import {
  applyTheme,
  readThemeMode,
  writeThemeMode,
  type ThemeMode,
} from '@/components/ui/theme-toggle';
import { mockUser } from '@/lib/mock';
import { cn } from '@/lib/utils';

const MODEL_KEY = 'synaptic-model';
const HYDE_KEY = 'synaptic-hyde';

const MODELS = [
  { value: 'gemma-4-E4B-it', label: 'Gemma 4 E4B — fast' },
  { value: 'gemma-4-26b-A4B-it', label: 'Gemma 4 26B A4B — thorough' },
];

function Card({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-5">
      <h2 className="text-sm font-medium text-foreground">{title}</h2>
      <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
      <div className="mt-4">{children}</div>
    </section>
  );
}

const THEME_OPTIONS: { value: ThemeMode; label: string; swatch: string }[] = [
  {
    value: 'light',
    label: 'Light',
    swatch: 'bg-white',
  },
  {
    value: 'dark',
    label: 'Dark',
    swatch: 'bg-neutral-900',
  },
  {
    value: 'system',
    label: 'System',
    swatch: 'bg-linear-to-r from-white to-neutral-900',
  },
];

export default function ProfilePage() {
  const [mode, setMode] = useState<ThemeMode>('system');
  const [model, setModel] = useState(MODELS[0].value);
  const [hyde, setHyde] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Mount-time sync of preferences persisted outside React (localStorage +
    // theme class). Intentional one-shot reads from external systems.
    /* eslint-disable react-hooks/set-state-in-effect -- intentional mount-time reads */
    setMode(readThemeMode());
    try {
      setModel(localStorage.getItem(MODEL_KEY) ?? MODELS[0].value);
      setHyde(localStorage.getItem(HYDE_KEY) === 'true');
    } catch {
      // Defaults are fine without storage
    }
    setMounted(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  function chooseTheme(next: ThemeMode) {
    setMode(next);
    applyTheme(next);
    writeThemeMode(next);
  }

  function chooseModel(next: string) {
    setModel(next);
    try {
      localStorage.setItem(MODEL_KEY, next);
    } catch {
      // Ignore
    }
  }

  function toggleHyde() {
    setHyde((prev) => {
      try {
        localStorage.setItem(HYDE_KEY, String(!prev));
      } catch {
        // Ignore
      }
      return !prev;
    });
  }

  return (
    <AppShell sidebar={{ currentTitle: 'Settings' }}>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl space-y-5 px-4 py-8">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-foreground">
              Settings
            </h1>
            <p className="mt-0.5 text-sm text-muted-foreground">
              Manage your account and generation defaults.
            </p>
          </div>

          <Card
            title="Profile"
            description="This information is visible to your workspace."
          >
            <div className="flex items-center gap-4">
              <span className="flex size-14 items-center justify-center rounded-[30%] bg-linear-to-br from-[#da7756] to-[#b85c3a] text-xl font-semibold text-white">
                {mockUser.initials}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <p className="truncate text-sm font-medium text-foreground">
                    {mockUser.name}
                  </p>
                  <span className="rounded-md border border-border px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                    {mockUser.plan}
                  </span>
                </div>
                <p className="truncate font-mono text-[11px] text-muted-foreground/80">
                  {mockUser.email} · Member since {mockUser.memberSince}
                </p>
              </div>
            </div>
          </Card>

          <Card
            title="Appearance"
            description="Choose how Synaptic looks on this device."
          >
            <div className="grid max-w-md grid-cols-3 gap-2">
              {THEME_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  onClick={() => chooseTheme(option.value)}
                  disabled={!mounted}
                  aria-pressed={mode === option.value}
                  className={cn(
                    'rounded-xl border bg-background p-2.5 text-left transition-[border-color,box-shadow] duration-150 hover:border-ring/50',
                    mounted && mode === option.value
                      ? 'border-ring ring-1 ring-ring/30'
                      : 'border-border',
                  )}
                >
                  <span
                    className={cn(
                      'block h-10 w-full rounded-md border border-border/60',
                      option.swatch,
                    )}
                  />
                  <span className="mt-2 block text-xs font-medium text-foreground">
                    {option.label}
                  </span>
                </button>
              ))}
            </div>
          </Card>

          <Card
            title="Generation defaults"
            description="Applied to new requests in this browser."
          >
            <div className="max-w-md space-y-4">
              <div>
                <label
                  htmlFor="model"
                  className="mb-1.5 block text-xs font-medium text-foreground"
                >
                  Model
                </label>
                <select
                  id="model"
                  value={model}
                  onChange={(e) => chooseModel(e.target.value)}
                  className="h-9 w-full cursor-pointer rounded-lg border border-input bg-transparent px-2.5 text-sm text-foreground outline-none transition-colors focus:border-ring/70"
                >
                  {MODELS.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-xs font-medium text-foreground">
                    Query condensation (HyDE)
                  </p>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    Rewrite the question before retrieval. Slower, better recall.
                  </p>
                </div>
                <button
                  role="switch"
                  aria-checked={hyde}
                  aria-label="Toggle HyDE"
                  onClick={toggleHyde}
                  className={cn(
                    'relative h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors duration-150',
                    hyde
                      ? 'bg-primary'
                      : 'bg-input',
                  )}
                >
                  <span
                    className={cn(
                      'absolute top-0.5 left-0.5 block size-4 rounded-full bg-white shadow-sm transition-transform duration-150',
                      hyde && 'translate-x-4',
                    )}
                  />
                </button>
              </div>
            </div>
          </Card>
        </div>
      </main>
    </AppShell>
  );
}
