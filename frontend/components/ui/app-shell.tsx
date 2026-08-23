'use client';

import { useState } from 'react';
import { AppSidebar, SidebarProvider } from './app-sidebar';

export function AppShell({
  children,
  sidebar,
}: {
  children: React.ReactNode;
  sidebar?: React.ComponentProps<typeof AppSidebar>;
}) {
  const [open, setOpen] = useState(false);

  return (
    <SidebarProvider value={{ open, setOpen }}>
      <div className="flex h-dvh overflow-hidden">
        <AppSidebar {...sidebar} />
        <div className="flex min-w-0 flex-1 flex-col">{children}</div>
      </div>
    </SidebarProvider>
  );
}
