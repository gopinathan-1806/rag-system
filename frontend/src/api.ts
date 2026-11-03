import type { ChatEvent, DocumentInfo, FitReport, SessionState } from "./types";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) throw await toError(res);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

async function toError(res: Response): Promise<ApiError> {
  let message = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (typeof body?.error === "string") message = body.error;
    else if (res.status === 422) message = "That input was not valid.";
  } catch {
    /* non-JSON error body: keep generic message */
  }
  return new ApiError(message, res.status);
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  createSession: () => request<SessionState>("/api/sessions", { method: "POST" }),
  getSession: (sid: string) => request<SessionState>(`/api/sessions/${sid}`),
  deleteSession: (sid: string) => request<void>(`/api/sessions/${sid}`, { method: "DELETE" }),
  loadDemo: (sid: string) => request<SessionState>(`/api/sessions/${sid}/demo`, { method: "POST" }),
  upload(sid: string, kind: "resume" | "jobs", file: File) {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentInfo>(`/api/sessions/${sid}/${kind}`, { method: "POST", body: form });
  },
  paste: (sid: string, kind: "resume" | "jobs", title: string, text: string) =>
    request<DocumentInfo>(`/api/sessions/${sid}/${kind}/text`, json({ title, text })),
  removeDocument: (sid: string, docId: string) =>
    request<void>(`/api/sessions/${sid}/documents/${docId}`, { method: "DELETE" }),
  fit: (sid: string, jobId: string, refresh = false) =>
    request<FitReport>(`/api/sessions/${sid}/jobs/${jobId}/fit?refresh=${refresh}`, { method: "POST" }),
  clearChat: (sid: string) => request<void>(`/api/sessions/${sid}/chat`, { method: "DELETE" }),
};

/** Parse a server-sent-event stream body into events. Exported for testing. */
export async function* parseSse(body: ReadableStream<Uint8Array>): AsyncGenerator<ChatEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data: "))
        .map((l) => l.slice(6))
        .join("\n");
      if (data) yield JSON.parse(data) as ChatEvent;
    }
  }
}

export async function* streamChat(sid: string, question: string, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const res = await fetch(`/api/sessions/${sid}/chat`, { ...json({ question }), signal });
  if (!res.ok || !res.body) throw await toError(res);
  yield* parseSse(res.body);
}
