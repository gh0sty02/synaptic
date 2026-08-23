'use client';

import { useRouter } from 'next/navigation';
import { ChevronUp, LogOut, UserRound } from 'lucide-react';
import { DropdownMenu } from 'radix-ui';
import { mockUser } from '@/lib/mock';

const itemClass =
  'flex h-8 cursor-pointer select-none items-center gap-2 rounded-lg px-2 text-[13px] text-foreground outline-none data-highlighted:bg-accent';

export function UserMenu() {
  const router = useRouter();

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          className="flex w-full items-center gap-2 rounded-lg p-1.5 text-left transition-colors outline-none hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring/30"
          aria-label="Open account menu"
        >
          <span className="flex size-6 shrink-0 items-center justify-center rounded-[30%] bg-linear-to-br from-[#da7756] to-[#b85c3a] text-[10px] font-semibold text-white">
            {mockUser.initials}
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-xs font-medium text-foreground">
              {mockUser.name}
            </span>
            <span className="block truncate font-mono text-[10px] text-muted-foreground/80">
              {mockUser.plan} plan
            </span>
          </span>
          <ChevronUp
            className="size-3.5 shrink-0 text-muted-foreground"
            aria-hidden="true"
          />
        </button>
      </DropdownMenu.Trigger>

        <DropdownMenu.Portal>
          <DropdownMenu.Content
            side="top"
            align="start"
            sideOffset={6}
            style={{ transformOrigin: 'var(--radix-dropdown-menu-content-transform-origin)' }}
            className="z-50 w-52 rounded-lg border border-border bg-popover p-1 duration-150 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-in-95"
          >
          <div className="px-2 py-1.5">
            <p className="truncate text-[13px] font-medium">{mockUser.name}</p>
            <p className="truncate font-mono text-[11px] text-muted-foreground/80">
              {mockUser.email}
            </p>
          </div>
          <DropdownMenu.Separator className="my-1 h-px bg-border" />
          <DropdownMenu.Item asChild>
            <a href="/profile" className={itemClass}>
              <UserRound className="size-3.5 text-muted-foreground" aria-hidden="true" />
              Profile settings
            </a>
          </DropdownMenu.Item>
          <DropdownMenu.Separator className="my-1 h-px bg-border" />
          <DropdownMenu.Item
            onSelect={() => router.push('/auth')}
            className={`${itemClass} text-destructive data-highlighted:bg-destructive/10`}
          >
            <LogOut className="size-3.5" aria-hidden="true" />
            Log out
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
