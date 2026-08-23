import { ShieldAlert } from 'lucide-react';

export function WarningBanner({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-2.5 rounded-xl border border-amber-500/30 bg-amber-500/8 px-3.5 py-2.5">
      <ShieldAlert
        className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400"
        aria-hidden="true"
      />
      <div className="space-y-0.5">
        <p className="text-[13px] font-medium leading-5 text-amber-700 dark:text-amber-400">
          Response withheld by output guardrail
        </p>
        <p className="text-xs leading-5 text-amber-700/80 dark:text-amber-400/75">
          {children}
        </p>
      </div>
    </div>
  );
}
