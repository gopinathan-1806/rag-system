import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import { api, streamChat } from "../api";
import { citationId, linkCitations } from "../citations";
import type { Message, SessionState, Source } from "../types";

const SUGGESTIONS = [
  "What skills am I missing for Job #1?",
  "How does my experience align with Job #2?",
  "Which job is the best fit for me, and why?",
  "Give me interview questions for Job #1 and how I should answer them.",
];

interface Props {
  state: SessionState;
  ready: boolean;
}

export function Chat({ state, ready }: Props) {
  const sid = state.session_id;
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [active, setActive] = useState<{ msg: number; id: string } | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);
  useEffect(
    () => () => {
      abort.current?.abort();
    },
    [],
  );

  function patch(id: number, fn: (m: Message) => Message) {
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
  }

  async function send(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    const userMsg: Message = { id: nextId.current++, role: "user", text: q };
    const botMsg: Message = { id: nextId.current++, role: "assistant", text: "", streaming: true };
    setMessages((ms) => [...ms, userMsg, botMsg]);
    setInput("");
    setBusy(true);
    setActive(null);
    abort.current = new AbortController();
    try {
      for await (const ev of streamChat(sid, q, abort.current.signal)) {
        if (ev.type === "sources") patch(botMsg.id, (m) => ({ ...m, sources: ev.sources }));
        else if (ev.type === "token") patch(botMsg.id, (m) => ({ ...m, text: m.text + ev.text }));
        else if (ev.type === "done")
          patch(botMsg.id, (m) => ({ ...m, text: ev.answer ?? m.text, cited: ev.cited, streaming: false }));
        else if (ev.type === "error") patch(botMsg.id, (m) => ({ ...m, text: ev.message, error: true, streaming: false }));
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError")
        patch(botMsg.id, (m) => ({ ...m, text: e instanceof Error ? e.message : "Something went wrong.", error: true, streaming: false }));
    } finally {
      patch(botMsg.id, (m) => ({ ...m, streaming: false }));
      setBusy(false);
    }
  }

  async function clear() {
    abort.current?.abort();
    await api.clearChat(sid).catch(() => undefined);
    setMessages([]);
    setActive(null);
  }

  const activeSource: Source | undefined = active
    ? messages.find((m) => m.id === active.msg)?.sources?.find((s) => s.id === active.id)
    : undefined;

  return (
    <div className="chat">
      <div className="chat-scroll">
        {messages.length === 0 ? (
          <Empty ready={ready} llm={state.llm_enabled} onPick={send} />
        ) : (
          messages.map((m) => (
            <Bubble key={m.id} m={m} onCite={(id) => setActive({ msg: m.id, id })} activeId={active?.msg === m.id ? active.id : null} />
          ))
        )}
        <div ref={bottom} />
      </div>

      {activeSource && <SourcePanel source={activeSource} onClose={() => setActive(null)} />}

      <form className="composer" onSubmit={(e) => { e.preventDefault(); void send(input); }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(input); } }}
          placeholder={ready ? "Ask about fit, skill gaps, or interview prep…" : "Add your resume and a job posting to begin"}
          disabled={!ready || busy}
          rows={1}
          maxLength={1000}
          aria-label="Your question"
        />
        <button className="btn primary" type="submit" disabled={!ready || busy || !input.trim()}>
          {busy ? "…" : "Ask"}
        </button>
        {messages.length > 0 && (
          <button type="button" className="btn ghost" onClick={clear} disabled={busy}>Clear</button>
        )}
      </form>
    </div>
  );
}

function Empty({ ready, llm, onPick }: { ready: boolean; llm: boolean; onPick: (q: string) => void }) {
  return (
    <div className="empty">
      <h2>Ask anything about your fit</h2>
      <p>Answers are grounded in your resume and the postings you add — every claim links back to its source passage.</p>
      {!llm && <div className="notice error">No LLM is configured on the server. Set <code>LLM_API_KEY</code> to enable answers.</div>}
      <div className="suggestions">
        {SUGGESTIONS.map((s) => (
          <button key={s} className="chip-btn" disabled={!ready || !llm} onClick={() => onPick(s)}>{s}</button>
        ))}
      </div>
    </div>
  );
}

function Bubble({ m, onCite, activeId }: { m: Message; onCite: (id: string) => void; activeId: string | null }) {
  if (m.role === "user") return <div className="bubble user">{m.text}</div>;
  const cited = new Set(m.cited ?? []);
  return (
    <div className={`bubble bot ${m.error ? "error" : ""}`}>
      {m.text === "" && m.streaming ? (
        <span className="typing" aria-label="Thinking"><i /><i /><i /></span>
      ) : (
        <Markdown
          components={{
            a({ href, children }) {
              const id = citationId(href);
              if (!id) return <span>{children}</span>; // never render model-supplied links
              const known = m.sources?.some((s) => s.id === id);
              return (
                <button type="button" className={`cite ${activeId === id ? "active" : ""}`} disabled={!known} onClick={() => onCite(id)}>
                  {id.replace("S", "")}
                </button>
              );
            },
          }}
        >
          {linkCitations(m.text)}
        </Markdown>
      )}
      {!m.streaming && !m.error && m.sources && m.sources.length > 0 && (
        <div className="bubble-foot">
          {cited.size > 0 ? `Grounded in ${cited.size} source${cited.size > 1 ? "s" : ""}` : "⚠ No citations in this answer — verify before relying on it"}
          <span> · {m.sources.length} passages retrieved</span>
        </div>
      )}
    </div>
  );
}

function SourcePanel({ source, onClose }: { source: Source; onClose: () => void }) {
  return (
    <div className="source-panel" role="region" aria-label="Source passage">
      <div className="source-head">
        <span>
          <b>[{source.id.replace("S", "")}]</b> {source.label} · {source.section}
        </span>
        <button className="icon-btn" onClick={onClose} aria-label="Close source">×</button>
      </div>
      <pre>{source.text}</pre>
      <div className="source-meta">{source.title} · semantic similarity {source.cosine.toFixed(2)}</div>
    </div>
  );
}
