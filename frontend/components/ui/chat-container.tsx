'use client';

import { useState, useRef, useEffect } from 'react';
import OpenAI from 'openai';
import { Brain, ChevronDown, ArrowUp, LoaderCircle } from 'lucide-react';
import ReactMarkdown, { type Components } from 'react-markdown';
import { Button } from './button';
import { cn } from '@/lib/utils';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy',
  dangerouslyAllowBrowser: true,
});

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

type ParsedAssistantMessage = {
  thinking: string;
  answer: string;
  isThinkingOpen: boolean;
};

function parseAssistantMessage(content: string): ParsedAssistantMessage {
  const thinkStart = content.indexOf('<think>');

  if (thinkStart === -1) {
    return { thinking: '', answer: content, isThinkingOpen: false };
  }

  const thinkingContentStart = thinkStart + '<think>'.length;
  const thinkEnd = content.indexOf('</think>', thinkingContentStart);

  if (thinkEnd === -1) {
    return {
      thinking: content.slice(thinkingContentStart).trim(),
      answer: content.slice(0, thinkStart).trim(),
      isThinkingOpen: true,
    };
  }

  return {
    thinking: content.slice(thinkingContentStart, thinkEnd).trim(),
    answer:
      `${content.slice(0, thinkStart)}${content.slice(thinkEnd + '</think>'.length)}`.trim(),
    isThinkingOpen: false,
  };
}

const markdownComponents: Components = {
  h1: ({ children }) => (
    <h1 className="text-xl font-semibold leading-7 text-foreground">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="pt-1 text-lg font-semibold leading-7 text-foreground">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="pt-1 text-base font-semibold leading-6 text-foreground">
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
      className="font-medium underline underline-offset-4"
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
    <pre className="overflow-x-auto rounded-xl border border-border bg-foreground p-4 text-xs leading-6 text-background">
      {children}
    </pre>
  ),
  code: ({ children, className }) => {
    const isBlockCode = className?.startsWith('language-');
    if (isBlockCode) {
      return <code className={cn('font-mono', className)}>{children}</code>;
    }
    return (
      <code className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-[0.9em] text-foreground">
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
        'flex shrink-0 items-center justify-center rounded-full bg-linear-to-br from-[#da7756] to-[#b85c3a] font-semibold text-white shadow-sm',
        size === 'lg' ? 'size-12 text-lg' : 'size-7 text-[11px]',
      )}
    >
      S
    </div>
  );
}

function AssistantMessage({ content }: { content: string }) {
  const { thinking, answer, isThinkingOpen } = parseAssistantMessage(content);

  return (
    <div className="flex gap-3">
      <AssistantAvatar />
      <div className="min-w-0 flex-1 space-y-3 pt-0.5">
        {thinking && (
          <details
            key={isThinkingOpen ? 'thinking-open' : 'thinking-closed'}
            className="group rounded-xl border border-border/60 bg-muted/40 px-3 py-2"
            open={isThinkingOpen || undefined}
          >
            <summary className="flex cursor-pointer list-none items-center gap-2 text-xs font-medium text-muted-foreground [&::-webkit-details-marker]:hidden">
              <Brain className="size-3.5" aria-hidden="true" />
              <span>{isThinkingOpen ? 'Thinking...' : 'Thought process'}</span>
              <ChevronDown
                className="ml-auto size-3.5 transition-transform group-open:rotate-180"
                aria-hidden="true"
              />
            </summary>
            <div className="mt-2 max-h-64 overflow-y-auto border-l border-border/60 pl-3 text-muted-foreground [&_.space-y-3]:space-y-1.5 [&_p]:text-xs [&_p]:leading-6 [&_li]:text-xs [&_li]:leading-6 [&_strong]:text-muted-foreground">
              <MarkdownText content={thinking} />
            </div>
          </details>
        )}
        {answer && <MarkdownText content={answer} />}
      </div>
    </div>
  );
}

function ChatMessage({ message }: { message: Message }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] whitespace-pre-wrap rounded-3xl bg-(--user-bubble) px-4 py-3 text-sm leading-7 text-foreground">
          {message.content}
        </div>
      </div>
    );
  }
  return <AssistantMessage content={message.content} />;
}

export const ChatContainer = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function onSubmitHandler() {
    if (!input.trim() || isStreaming) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
    };
    const assistantId = crypto.randomUUID();

    setInput('');
    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantId, role: 'assistant', content: '' },
    ]);
    setIsStreaming(true);

    try {
      const stream = await client.chat.completions.create({
        model: 'synaptic',
        messages: [...messages, userMessage].map((m) => ({
          role: m.role,
          content: m.content,
        })),
        stream: true,
      });

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
    } finally {
      setIsStreaming(false);
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      {/* Messages / empty state */}
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 px-4">
            <AssistantAvatar size="lg" />
            <div className="text-center">
              <h2 className="text-2xl font-semibold tracking-tight text-foreground">
                How can I help you today?
              </h2>
              <p className="mt-1.5 text-sm text-muted-foreground">
                Ask anything about your indexed StackOverflow knowledge base.
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-2xl space-y-6 px-4 py-8">
            {messages.map((m) => (
              <ChatMessage key={m.id} message={m} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="w-full px-4 pb-5 pt-2">
        <div className="mx-auto max-w-2xl">
          <div className="rounded-2xl border border-border bg-card shadow-sm transition-shadow focus-within:shadow-md">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  void onSubmitHandler();
                }
              }}
              placeholder="Message Synaptic..."
              rows={1}
              disabled={isStreaming}
              className="w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-sm leading-6 outline-none placeholder:text-muted-foreground disabled:opacity-60"
            />
            <div className="flex items-center justify-between px-3 pb-3">
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                {isStreaming && (
                  <>
                    <LoaderCircle
                      className="size-3 animate-spin"
                      aria-hidden="true"
                    />
                    Responding…
                  </>
                )}
              </span>
              <Button
                onClick={() => void onSubmitHandler()}
                disabled={isStreaming || !input.trim()}
                size="icon-sm"
                className="rounded-full"
                aria-label="Send message"
              >
                <ArrowUp className="size-4" aria-hidden="true" />
              </Button>
            </div>
          </div>
          <p className="mt-2 text-center text-xs text-muted-foreground/60">
            Synaptic can make mistakes. Verify important information.
          </p>
        </div>
      </div>
    </div>
  );
};
