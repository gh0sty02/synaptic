import { ExternalLink } from 'lucide-react';
import type { Citation } from '@/lib/mock';

export function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1.5">
      {citations.map((c) => (
        <a
          key={c.index}
          href={c.url}
          target="_blank"
          rel="noreferrer"
          title={`${c.title} · relevance ${c.score.toFixed(2)}`}
          className="group/cite flex max-w-[280px] items-center gap-1.5 rounded-lg border border-border bg-card px-2 py-1 font-mono text-[11px] text-muted-foreground transition-colors duration-150 hover:border-ring/50 hover:text-foreground"
        >
          <span>{c.index}</span>
          <span className="truncate">{c.domain}</span>
          <ExternalLink
            className="size-3 shrink-0 opacity-0 transition-opacity group-hover/cite:opacity-100"
            aria-hidden="true"
          />
        </a>
      ))}
    </div>
  );
}
