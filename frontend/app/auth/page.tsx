'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

function GoogleMark() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden="true">
      <path
        fill="#EA4335"
        d="M12 5.04c1.62 0 3.06.56 4.2 1.64l3.12-3.12C17.46 1.8 14.96.75 12 .75 7.6.75 3.8 3.27 1.96 6.96l3.66 2.84C6.5 7.09 9 5.04 12 5.04Z"
      />
      <path
        fill="#4285F4"
        d="M23.25 12.27c0-.79-.07-1.54-.19-2.27H12v4.51h6.47c-.29 1.48-1.14 2.73-2.41 3.57l3.68 2.85c2.15-1.99 3.51-4.92 3.51-8.66Z"
      />
      <path
        fill="#FBBC05"
        d="M5.62 14.2a7.2 7.2 0 0 1 0-4.4L1.96 6.96a11.26 11.26 0 0 0 0 10.08l3.66-2.84Z"
      />
      <path
        fill="#34A853"
        d="M12 23.25c3.04 0 5.58-1 7.44-2.72l-3.68-2.85c-1 .69-2.3 1.1-3.76 1.1-3 0-5.5-2.05-6.38-4.78l-3.66 2.84c1.84 3.69 5.64 6.41 10.04 6.41Z"
      />
    </svg>
  );
}

function GitHubMark() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.55 0-.27-.01-1.17-.02-2.12-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.76 2.69 1.25 3.35.96.1-.75.4-1.25.72-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.18-3.09-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.24 2.76.12 3.05.74.81 1.18 1.83 1.18 3.09 0 4.42-2.7 5.39-5.26 5.67.41.36.78 1.06.78 2.14 0 1.54-.02 2.79-.02 3.17 0 .31.21.67.8.55A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"
      />
    </svg>
  );
}

type FormErrors = { email?: string; password?: string };

export default function AuthPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});

  function signIn(e: React.FormEvent) {
    e.preventDefault();

    const next: FormErrors = {};
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      next.email = 'Enter a valid email address.';
    }
    if (password.length < 8) {
      next.password = 'Password must be at least 8 characters.';
    }
    setErrors(next);

    if (Object.keys(next).length === 0) {
      router.push('/');
    }
  }

  const inputClass =
    'h-10 w-full rounded-lg border border-input bg-transparent px-3 text-sm text-foreground outline-none transition-colors duration-150 placeholder:text-muted-foreground/60 focus:border-ring/70';

  return (
    <div className="relative grid min-h-dvh place-items-center bg-background px-4">
      <Button
        asChild
        variant="ghost"
        size="icon-sm"
        className="absolute left-4 top-4"
      >
        <Link href="/" aria-label="Back to chat">
          <ArrowLeft className="size-4" aria-hidden="true" />
        </Link>
      </Button>

      <div className="w-full max-w-sm">
        <div className="mb-8 flex flex-col items-center">
          <span className="rise-in flex size-11 items-center justify-center rounded-md bg-linear-to-br from-[#da7756] to-[#b85c3a] text-lg font-semibold text-white">
            S
          </span>
          <h1 className="rise-in mt-4 text-xl font-semibold tracking-tight text-foreground [animation-delay:60ms]">
            Sign in to Synaptic
          </h1>
          <p className="rise-in mt-1 text-sm text-muted-foreground [animation-delay:110ms]">
            Use your work account to continue.
          </p>
        </div>

        <form onSubmit={signIn} noValidate className="space-y-4">
          <div>
            <label
              htmlFor="email"
              className="mb-1.5 block text-xs font-medium text-foreground"
            >
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
              className={`${inputClass} ${errors.email ? 'border-destructive' : ''}`}
            />
            {errors.email && (
              <p className="mt-1.5 text-xs text-destructive">{errors.email}</p>
            )}
          </div>

          <div>
            <label
              htmlFor="password"
              className="mb-1.5 block text-xs font-medium text-foreground"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className={`${inputClass} ${errors.password ? 'border-destructive' : ''}`}
            />
            {errors.password && (
              <p className="mt-1.5 text-xs text-destructive">
                {errors.password}
              </p>
            )}
          </div>

          <Button type="submit" className="h-9 w-full">
            Continue with email
          </Button>
        </form>

        <div className="my-6 flex items-center gap-3">
          <span className="fade-divider-x flex-1" />
          <span className="font-mono text-[11px] text-muted-foreground">or</span>
          <span className="fade-divider-x flex-1" />
        </div>

        <div className="space-y-2">
          <Button variant="outline" className="h-9 w-full gap-2">
            <GoogleMark />
            Continue with Google
          </Button>
          <Button variant="outline" className="h-9 w-full gap-2">
            <GitHubMark />
            Continue with GitHub
          </Button>
        </div>

        <p className="mt-6 text-center text-xs text-muted-foreground">
          New here?{' '}
          <Link
            href="/"
            className="font-medium text-primary underline decoration-primary/40 underline-offset-4 hover:decoration-primary"
          >
            Try the demo without an account
          </Link>
        </p>
        <p className="mt-3 text-center font-mono text-[10px] tracking-wide text-muted-foreground/50">
          By continuing you agree to the Terms of Service.
        </p>
      </div>
    </div>
  );
}
