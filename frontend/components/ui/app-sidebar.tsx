'use client';

import { createContext, useContext } from 'react';
import { Plus, X } from 'lucide-react';
import type { MockConversation } from '@/lib/mock';
import {
  GROUP_LABELS,
  mockConversations,
} from '@/lib/mock';
import { UserMenu } from './user-menu';
import { cn } from '@/lib/utils';

type SidebarProps = {
  currentTitle?: string;
  activeId?: string | null;
  onNewChat?: () => void;
  onSelectConversation?: (conversation: MockConversation) => void;
};

export function useSidebar() {
  return useContext(SidebarContext);
}

const SidebarContext = createContext<{ open: boolean; setOpen: (open: boolean) => void }>({
  open: false,
  setOpen: () => {},
});

export const SidebarProvider = SidebarContext.Provider;

function ConversationItem({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'relative flex w-full cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] transition-colors duration-150 ease-out',
        active
          ? 'bg-accent font-medium text-foreground'
          : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
        active && 'pl-4',
      )}
    >
      {active && (
        <span
          className="absolute top-1/2 left-1.5 size-1.5 -translate-y-1/2 rounded-full bg-primary"
          aria-hidden="true"
        />
      )}
      <span className="truncate">{label}</span>
    </button>
  );
}

export function AppSidebar({
  currentTitle = 'New chat',
  activeId = null,
  onNewChat,
  onSelectConversation,
}: SidebarProps) {
  const { open, setOpen } = useSidebar();

  const groups = ['today', 'yesterday', 'week'] as const;

  return (
    <>
      {/* Mobile backdrop */}
      <div
        onClick={() => setOpen(false)}
        className={cn(
          'fixed inset-0 z-30 bg-black/40 backdrop-blur-[2px] transition-opacity md:hidden',
          open ? 'opacity-100' : 'pointer-events-none opacity-0',
        )}
        aria-hidden="true"
      />

      <aside
        className={cn(
          'z-40 flex w-60 shrink-0 flex-col border-r border-border/60 bg-background transition-transform duration-200 max-md:fixed max-md:inset-y-0 max-md:left-0',
          open ? 'max-md:translate-x-0' : 'max-md:-translate-x-full',
        )}
        aria-label="Chat history"
      >
        <div className="flex h-12 shrink-0 items-center justify-between px-3">
          <span className="text-sm font-semibold tracking-tight text-foreground">
            Synaptic
          </span>
          <button
            onClick={() => setOpen(false)}
            className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground md:hidden"
            aria-label="Close sidebar"
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        </div>

        <div className="px-2 pb-1">
          <button
            onClick={() => {
              onNewChat?.();
              setOpen(false);
            }}
            className="flex h-8 w-full items-center gap-2 rounded-lg border border-border bg-transparent px-2.5 text-[13px] font-medium text-foreground transition-colors duration-150 hover:border-ring/50 hover:bg-accent"
          >
            <Plus className="size-3.5 text-muted-foreground" aria-hidden="true" />
            New chat
          </button>
        </div>

        <nav className="flex-1 space-y-4 overflow-y-auto px-2 py-3">
          {groups.map((group) => {
            const items = mockConversations.filter((c) => c.group === group);
            if (items.length === 0) return null;

            return (
              <div key={group}>
                <p className="mb-1 px-2 font-mono text-[11px] font-medium uppercase tracking-wider text-muted-foreground/60">
                  {GROUP_LABELS[group]}
                </p>
                <div className="space-y-0.5">
                  {group === 'today' && (
                    <ConversationItem
                      label={currentTitle}
                      active={activeId === null || activeId === 'current'}
                      onClick={() => setOpen(false)}
                    />
                  )}
                  {items.map((conversation) => (
                    <ConversationItem
                      key={conversation.id}
                      label={conversation.title}
                      active={activeId === conversation.id}
                      onClick={() => {
                        onSelectConversation?.(conversation);
                        setOpen(false);
                      }}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </nav>

        <div className="shrink-0 border-t border-border/60 p-2">
          <UserMenu />
        </div>
      </aside>
    </>
  );
}
