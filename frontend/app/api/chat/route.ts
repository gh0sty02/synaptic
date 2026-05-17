import { createDataStreamResponse } from 'ai';

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

export async function POST(request: Request) {
  const { messages, id } = await request.json();
  const lastMessage = messages[messages.length - 1];
  const query = lastMessage.parts?.find(
    (p: { type: string }) => p.type === 'text'
  )?.text as string;

  const fastApiRes = await fetch(`${FASTAPI_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: id }),
  });

  if (!fastApiRes.ok || !fastApiRes.body) {
    return new Response('FastAPI error', { status: 502 });
  }

  return createDataStreamResponse({
    execute: async (writer) => {
      const reader = fastApiRes.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split('\n\n');
        buffer = frames.pop() ?? '';

        for (const frame of frames) {
          if (!frame.startsWith('data: ')) continue;
          try {
            const payload = JSON.parse(frame.slice(6));
            if (
              payload.type === 'token' &&
              typeof payload.content === 'string'
            ) {
              writer.writeTextDelta(payload.content);
            }
          } catch {
            // malformed frame — skip
          }
        }
      }
    },
  });
}
