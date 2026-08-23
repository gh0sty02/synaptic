'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import OpenAI from 'openai';
import {
  Brain,
  ChevronDown,
  ArrowUp,
  Square,
  Copy,
  Check,
  RotateCcw,
  LoaderCircle,
  SquarePen,
  PanelLeft,
} from 'lucide-react';
import ReactMarkdown, { type Components } from 'react-markdown';
import { Button } from './button';
import { ThemeToggle } from './theme-toggle';
import { WarningBanner } from './warning-banner';
import { CitationList } from './citations';
import { AppShell } from './app-shell';
import { useSidebar } from './app-sidebar';
import { mockCitations, type Citation, type MockConversation } from '@/lib/mock';
import { cn } from '@/lib/utils';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy',
  dangerouslyAllowBrowser: true,
});

const RETRACTION_MARKER = '[Response retracted';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  error?: boolean;
  citations?: Citation[];
};

type ParsedAssistantMessage = {
  thinking: string;
  answer: string;
  isThinkingOpen: boolean;
  retracted: boolean;
};

const STARTER_PROMPTS = [
  {
    title: 'Why does my React effect fire twice?',
    hint: 'StrictMode behavior in development',
  },
  {
    title: 'Postgres index not being used',
    hint: 'Common query planner pitfalls',
  },
  {
    title: 'Fix circular imports in Python',
    hint: 'Restructuring modules cleanly',
  },
  {
    title: 'Git rebase vs merge',
    hint: 'When each workflow is safer',
  },
];

function parseAssistantMessage(content: string): ParsedAssistantMessage {
  const retracted = content.includes(RETRACTION_MARKER);
  const cleaned = content.replaceAll(/ ?\[Response retracted[^\]]*\]/g, '');

  const thinkStart = cleaned.indexOf('<think>');

  if (thinkStart === -1) {
    return { thinking: '', answer: cleaned, isThinkingOpen: false, retracted };
  }

  const thinkingContentStart = thinkStart + '<think>'.length;
  const thinkEnd = cleaned.indexOf('</think>', thinkingContentStart);

  if (thinkEnd === -1) {
    return {
      thinking: cleaned.slice(thinkingContentStart).trim(),
      answer: cleaned.slice(0, thinkStart).trim(),
      isThinkingOpen: true,
      retracted,
    };
  }

  return {
    thinking: cleaned.slice(thinkingContentStart, thinkEnd).trim(),
    answer: `${cleaned.slice(0, thinkStart)}${cleaned.slice(thinkEnd + '</think>'.length)}`.trim(),
    isThinkingOpen: false,
    retracted,
  };
}

const markdownComponents: Components = {
  h1: ({ children }) => (
    <h1 className="text-xl font-semibold tracking-tight text-foreground">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="pt-1 text-lg font-semibold tracking-tight text-foreground">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="pt-1 text-base font-semibold tracking-tight text-foreground">
      {children}
    </h3>
  ),
  p: ({ children }) => <p className="leading-7">{children}</p>,
  ul: ({ children }) => (
    <ul className="list-disc space-y-1 pl-5 leading-7">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal space-y-1 pl-5 leading-7">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-1">{children}</li>,
  strong: ({ children }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="font-medium text-muted-foreground underline decoration-muted-foreground/40 underline-offset-4 transition-colors hover:text-foreground hover:decoration-foreground/60"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-border pl-3 text-muted-foreground">
      {children}
    </blockquote>
  ),
  pre: ({ children }) => (
    <pre className="overflow-x-auto rounded-lg border border-border/70 bg-muted/60 p-4 text-xs leading-6">
      {children}
    </pre>
  ),
  code: ({ children, className }) => {
    const isBlockCode = className?.startsWith('language-');
    if (isBlockCode) {
      return <code className={cn('font-mono', className)}>{children}</code>;
    }
    return (
      <code className="rounded-md border border-border/50 bg-muted px-1.5 py-0.5 font-mono text-[0.9em] text-foreground">
        {children}
      </code>
    );
  },
};

function MarkdownText({ content }: { content: string }) {
  return (
    <div className="space-y-3 text-sm leading-7 text-foreground">
      <ReactMarkdown components={markdownComponents}>{content}</ReactMarkdown>
    </div>
  );
}

function AssistantAvatar({ size = 'sm' }: { size?: 'sm' | 'lg' }) {
  return (
    <div
      className={cn(
        'flex shrink-0 items-center justify-center rounded-md bg-linear-to-br from-[#da7756] to-[#b85c3a] font-semibold text-white',
        size === 'lg'
          ? 'size-12 text-lg shadow-lg shadow-primary/20'
          : 'size-6 text-[10px]',
      )}
    >
      S
    </div>
  );
}

function TypingIndicator() {
  return (
    <span
      className="flex items-center gap-1 py-2.5"
      aria-label="Synaptic is thinking"
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="typing-dot size-1.5 rounded-full bg-muted-foreground/70"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </span>
  );
}

function AssistantMessage({
  message,
  isStreaming,
  canRegenerate,
  onRegenerate,
}: {
  message: Message;
  isStreaming: boolean;
  canRegenerate: boolean;
  onRegenerate: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const { thinking, answer, isThinkingOpen, retracted } =
    parseAssistantMessage(message.content);

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard unavailable — nothing sensible to do in-place
    }
  }

  if (message.error) {
    return (
      <div className="flex gap-3">
        <AssistantAvatar />
        <div className="flex-1 pt-0.5">
          <div className="rounded-xl border border-destructive/25 bg-destructive/8 px-3.5 py-2.5">
            <p className="text-sm leading-6 text-destructive">
              {message.content}
            </p>
            {canRegenerate && (
              <button
                onClick={onRegenerate}
                className="mt-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
              >
                <RotateCcw className="size-3" aria-hidden="true" />
                Try again
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  const showTyping = isStreaming && !message.content;

  return (
    <div className="group/msg flex gap-3">
      <AssistantAvatar />
      <div className="min-w-0 flex-1 space-y-3 pt-0.5">
        {retracted && !isStreaming && (
          <WarningBanner>
            This answer was withheld because it triggered the output guardrail.
            Rephrase the question or contact an administrator if you believe this
            is a mistake.
          </WarningBanner>
        )}
        {thinking && (
          <details
            key={isThinkingOpen ? 'thinking-open' : 'thinking-closed'}
            className="group/thinking rounded-lg border border-border/60 bg-muted/40 px-3 py-2"
            open={isThinkingOpen || undefined}
          >
            <summary className="flex cursor-pointer list-none items-center gap-2 font-mono text-[11px] font-medium text-muted-foreground [&::-webkit-details-marker]:hidden">
              <Brain className="size-3.5" aria-hidden="true" />
              <span>
                {isStreaming && isThinkingOpen ? 'Thinking…' : 'Thought process'}
              </span>
              <ChevronDown
                className="ml-auto size-3.5 transition-transform group-open/thinking:rotate-180"
                aria-hidden="true"
              />
            </summary>
            <div className="mt-2 max-h-64 overflow-y-auto border-l border-border/60 pl-3 [&_.space-y-3]:space-y-1.5 [&_p]:text-xs [&_p]:leading-6 [&_li]:text-xs [&_li]:leading-6 [&_strong]:text-muted-foreground">
              <MarkdownText content={thinking} />
            </div>
          </details>
        )}
        {showTyping ? (
          <TypingIndicator />
        ) : (
          answer && <MarkdownText content={answer} />
        )}
        {!isStreaming && message.citations && (
          <CitationList citations={message.citations} />
        )}
        {!isStreaming && answer && (
          <div className="-mt-1 flex items-center gap-0.5 opacity-0 transition-opacity duration-200 group-hover/msg:opacity-100 focus-within:opacity-100">
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => void copyAnswer()}
              aria-label={copied ? 'Copied' : 'Copy answer'}
            >
              {copied ? (
                <Check className="size-3 text-primary" aria-hidden="true" />
              ) : (
                <Copy className="size-3" aria-hidden="true" />
              )}
            </Button>
            {canRegenerate && (
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={onRegenerate}
                aria-label="Regenerate response"
              >
                <RotateCcw className="size-3" aria-hidden="true" />
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ChatMessage({
  message,
  isStreaming,
  canRegenerate,
  onRegenerate,
}: {
  message: Message;
  isStreaming: boolean;
  canRegenerate: boolean;
  onRegenerate: () => void;
}) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[78%] whitespace-pre-wrap rounded-lg border border-border/60 bg-(--user-bubble) px-3.5 py-2.5 text-sm leading-7 text-foreground">
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <AssistantMessage
      message={message}
      isStreaming={isStreaming}
      canRegenerate={canRegenerate}
      onRegenerate={onRegenerate}
    />
  );
}

function EmptyState({ onPrompt }: { onPrompt: (prompt: string) => void }) {
  return (
    <div className="relative flex h-full flex-col items-center justify-center px-4 pb-16">
      <AssistantAvatar size="lg" />
      <h1 className="rise-in mt-6 text-center text-[30px] font-semibold leading-tight tracking-tighter text-foreground md:text-[34px]">
        <span className="text-muted-foreground/40">How can I </span>
        help you
        <span className="text-muted-foreground/40"> today?</span>
      </h1>
      <p className="rise-in mt-2.5 max-w-sm text-center text-sm leading-6 text-muted-foreground [animation-delay:60ms]">
        Ask anything about your indexed StackOverflow knowledge base.
      </p>
      <div className="mt-10 grid w-full max-w-lg grid-cols-1 gap-2 sm:grid-cols-2">
        {STARTER_PROMPTS.map((prompt, i) => (
          <button
            key={prompt.title}
            onClick={() => onPrompt(prompt.title)}
            style={{ animationDelay: `${120 + i * 55}ms` }}
            className="rise-in rounded-lg border border-border bg-card px-3.5 py-3 text-left transition-[border-color,background-color] duration-150 ease-out hover:border-ring/50 hover:bg-accent"
          >
            <p className="text-[13px] font-medium text-foreground">
              {prompt.title}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">{prompt.hint}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

export const ChatContainer = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const sessionId = useRef(crypto.randomUUID());
  const abortRef = useRef<AbortController | null>(null);
  const stoppedRef = useRef(false);
  const nearBottomRef = useRef(true);
  const { setOpen } = useSidebar();

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  useEffect(() => {
    if (nearBottomRef.current) {
      bottomRef.current?.scrollIntoView({
        behavior: isStreaming ? 'auto' : 'smooth',
      });
    }
  }, [messages, isStreaming]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    nearBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 96;
  }, []);

  function stop() {
    stoppedRef.current = true;
    abortRef.current?.abort();
  }

  function startNewChat() {
    stop();
    setInput('');
    setMessages([]);
    setActiveConversationId(null);
    sessionId.current = crypto.randomUUID();
    nearBottomRef.current = true;
  }

  function loadConversation(conversation: MockConversation) {
    stop();
    setActiveConversationId(conversation.id);
    sessionId.current = crypto.randomUUID();
    nearBottomRef.current = true;

    setMessages(
      conversation.messages.map((m, i) => ({
        id: `${conversation.id}-${i}`,
        role: m.role,
        content: m.content,
        ...(m.role === 'assistant' && !m.retracted
          ? { citations: mockCitations(`${conversation.id}-${i}`) }
          : {}),
      })),
    );
  }

  async function runCompletion(history: Message[], assistantId: string) {
    const controller = new AbortController();
    abortRef.current = controller;
    stoppedRef.current = false;

    try {
      // Spread keeps session_id past the SDK's strict body typing
      const extraParams = { session_id: sessionId.current };
      const stream = await client.chat.completions.create(
        {
          model: 'synaptic',
          messages: history.map((m) => ({ role: m.role, content: m.content })),
          stream: true,
          ...extraParams,
        },
        { signal: controller.signal },
      );

      for await (const chunk of stream) {
        const token = chunk.choices[0]?.delta?.content ?? '';
        if (token) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m,
            ),
          );
        }
      }

      // Placeholder until the backend emits citations in the stream payload
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && !m.error && !m.content.includes(RETRACTION_MARKER)
            ? { ...m, citations: mockCitations(assistantId) }
            : m,
        ),
      );
    } catch (err) {
      if (stoppedRef.current) return;

      const detail =
        err instanceof Error && err.message
          ? err.message
          : 'The request could not be completed.';
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, error: true, content: `Request failed: ${detail}` }
            : m,
        ),
      );
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setIsStreaming(false);
    }
  }

  async function send(prompt?: string) {
    const text = prompt ?? input;
    if (!text.trim() || isStreaming) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
    };
    const assistantId = crypto.randomUUID();
    const history = [...messages, userMessage];

    setInput('');
    setActiveConversationId((id) => (id === null ? null : 'current'));
    nearBottomRef.current = true;
    setMessages([...history, { id: assistantId, role: 'assistant', content: '' }]);
    setIsStreaming(true);

    await runCompletion(history, assistantId);
  }

  async function regenerate() {
    if (isStreaming) return;

    let lastUserIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserIndex = i;
        break;
      }
    }
    if (lastUserIndex === -1) return;

    const history = messages.slice(0, lastUserIndex + 1);
    const assistantId = crypto.randomUUID();

    nearBottomRef.current = true;
    setMessages([...history, { id: assistantId, role: 'assistant', content: '' }]);
    setIsStreaming(true);

    await runCompletion(history, assistantId);
  }

  const isEmpty = messages.length === 0;
  const lastMessage = messages[messages.length - 1];
  const lastAssistantId =
    lastMessage?.role === 'assistant' ? lastMessage.id : undefined;

  const firstUserMessage = messages.find((m) => m.role === 'user');
  const currentTitle = firstUserMessage
    ? firstUserMessage.content.length > 34
      ? `${firstUserMessage.content.slice(0, 34)}…`
      : firstUserMessage.content
    : 'New chat';

  return (
    <AppShell
      sidebar={{
        currentTitle,
        activeId: activeConversationId,
        onNewChat: startNewChat,
        onSelectConversation: loadConversation,
      }}
    >
      {/* Header */}
      <header className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-border/60 px-4">
        <div className="flex min-w-0 items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            className="md:hidden"
            onClick={() => setOpen(true)}
            aria-label="Open sidebar"
          >
            <PanelLeft className="size-4" aria-hidden="true" />
          </Button>
          <span className="truncate text-sm font-medium tracking-tight text-muted-foreground">
            {currentTitle}
          </span>
        </div>
        <div className="flex items-center gap-0.5">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={startNewChat}
            aria-label="Start new chat"
          >
            <SquarePen className="size-4" aria-hidden="true" />
          </Button>
          <ThemeToggle />
        </div>
      </header>

      {/* Messages / empty state */}
      <main
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto"
      >
        {isEmpty ? (
          <EmptyState onPrompt={(prompt) => void send(prompt)} />
        ) : (
          <div className="mx-auto max-w-2xl space-y-7 px-4 py-8">
            {messages.map((m) => (
              <div key={m.id} className="rise-in">
                <ChatMessage
                  message={m}
                  isStreaming={isStreaming && m.id === lastAssistantId}
                  canRegenerate={!isStreaming && m.id === lastAssistantId}
                  onRegenerate={() => void regenerate()}
                />
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      {/* Composer */}
      <div className="w-full shrink-0 px-4 pb-4 pt-2">
        <div className="mx-auto max-w-2xl">
          <div className="rounded-lg border border-input bg-card transition-colors duration-200 ease-out focus-within:border-ring/70">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder="Ask Synaptic…"
              rows={1}
              className="w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-sm leading-6 outline-none placeholder:text-muted-foreground/70"
            />
            <div className="flex items-center justify-between px-3 pb-3">
              <span className="flex h-5 items-center gap-1.5 font-mono text-[11px] text-muted-foreground/80">
                {isStreaming ? (
                  <>
                    <LoaderCircle
                      className="size-3 animate-spin [animation-duration:0.7s]"
                      aria-hidden="true"
                    />
                    Responding…
                  </>
                ) : (
                  <span className="hidden sm:block">
                    Enter to send · Shift+Enter for a newline
                  </span>
                )}
              </span>
              {isStreaming ? (
                <Button
                  variant="secondary"
                  size="icon-sm"
                  onClick={stop}
                  aria-label="Stop generating"
                  className="rounded-full"
                >
                  <Square className="size-3.5 fill-current" aria-hidden="true" />
                </Button>
              ) : (
                <Button
                  onClick={() => void send()}
                  disabled={!input.trim()}
                  size="icon-sm"
                  className="rounded-full"
                  aria-label="Send message"
                >
                  <ArrowUp className="size-4" aria-hidden="true" />
                </Button>
              )}
            </div>
          </div>
          <p className="mt-2 text-center font-mono text-[11px] text-muted-foreground/60">
            Synaptic can make mistakes. Verify important information.
          </p>
        </div>
      </div>
    </AppShell>
  );
};
