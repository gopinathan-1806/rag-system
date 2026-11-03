import { useRef, useState } from "react";
import { api } from "../api";
import type { DocumentInfo, SessionState } from "../types";

interface Props {
  state: SessionState;
  onChange: (s: SessionState) => void;
  onReset: () => void;
}

export function Sidebar({ state, onChange, onReset }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pasting, setPasting] = useState<"resume" | "jobs" | null>(null);
  const sid = state.session_id;

  async function run(label: string, fn: () => Promise<unknown>) {
    setBusy(label);
    setError(null);
    try {
      await fn();
      onChange(await api.getSession(sid));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setBusy(null);
    }
  }

  const upload = (kind: "resume" | "jobs", files: FileList | null) =>
    files &&
    run(`Indexing ${files.length > 1 ? `${files.length} files` : files[0].name}…`, async () => {
      for (const f of Array.from(files)) await api.upload(sid, kind, f);
    });

  return (
    <aside className="sidebar" aria-label="Documents">
      <div className="brand">
        <span className="logo" aria-hidden>◆</span>
        <div>
          <h1>Career Copilot</h1>
          <p>Resume ⇄ job fit, with receipts</p>
        </div>
      </div>

      {busy && <div className="notice busy" role="status">{busy}</div>}
      {error && <div className="notice error" role="alert">{error}</div>}

      <Section title="Your resume">
        {state.resume ? (
          <DocCard doc={state.resume} onRemove={() => run("Removing…", () => api.removeDocument(sid, state.resume!.id))} />
        ) : (
          <DropZone label="Upload resume" hint="PDF, DOCX, TXT" onFiles={(f) => upload("resume", f)} />
        )}
        {!state.resume && <PasteLink onClick={() => setPasting("resume")} />}
      </Section>

      <Section title={`Job postings (${state.jobs.length}/5)`}>
        {state.jobs.map((j) => (
          <DocCard key={j.id} doc={j} onRemove={() => run("Removing…", () => api.removeDocument(sid, j.id))} />
        ))}
        {state.jobs.length < 5 && (
          <DropZone label="Add job posting" hint="PDF, DOCX, TXT — multiple allowed" multiple onFiles={(f) => upload("jobs", f)} />
        )}
        {state.jobs.length < 5 && <PasteLink onClick={() => setPasting("jobs")} />}
      </Section>

      <div className="sidebar-footer">
        {!state.resume && state.jobs.length === 0 && (
          <button className="btn primary block" disabled={!!busy} onClick={() => run("Loading demo…", () => api.loadDemo(sid))}>
            Try with demo data
          </button>
        )}
        <button className="btn ghost block" onClick={onReset} title="Deletes all documents and chat from the server">
          Delete my data &amp; start over
        </button>
        <p className="privacy">Documents live in server memory only and expire after 2 hours.</p>
      </div>

      {pasting && (
        <PasteDialog
          kind={pasting}
          onClose={() => setPasting(null)}
          onSubmit={(title, text) =>
            run("Indexing…", () => api.paste(sid, pasting, title, text)).then(() => setPasting(null))
          }
        />
      )}
    </aside>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="section">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function DocCard({ doc, onRemove }: { doc: DocumentInfo; onRemove: () => void }) {
  return (
    <div className="doc">
      <div className="doc-head">
        <span className={`tag ${doc.kind}`}>{doc.label}</span>
        <button className="icon-btn" onClick={onRemove} aria-label={`Remove ${doc.label}`}>×</button>
      </div>
      <div className="doc-title" title={doc.filename}>{doc.title}</div>
      <div className="doc-meta">{doc.chunks} passages · {doc.chars.toLocaleString()} chars</div>
      {doc.warnings.map((w) => (
        <div key={w} className="doc-warning">⚠ {w}</div>
      ))}
    </div>
  );
}

function DropZone({ label, hint, multiple, onFiles }: { label: string; hint: string; multiple?: boolean; onFiles: (f: FileList | null) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      className={`drop ${over ? "over" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); onFiles(e.dataTransfer.files); }}
    >
      <strong>{label}</strong>
      <span>{hint}</span>
      <input ref={input} type="file" hidden multiple={multiple} accept=".pdf,.docx,.txt,.md"
        onChange={(e) => { onFiles(e.target.files); e.target.value = ""; }} />
    </div>
  );
}

const PasteLink = ({ onClick }: { onClick: () => void }) => (
  <button className="link" onClick={onClick}>or paste text</button>
);

function PasteDialog({ kind, onClose, onSubmit }: { kind: "resume" | "jobs"; onClose: () => void; onSubmit: (title: string, text: string) => void }) {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal onClick={(e) => e.stopPropagation()}>
        <h3>Paste {kind === "resume" ? "your resume" : "a job posting"}</h3>
        {kind === "jobs" && (
          <input placeholder="Title (optional, e.g. “Backend Engineer — Acme”)" value={title} maxLength={120} onChange={(e) => setTitle(e.target.value)} />
        )}
        <textarea autoFocus placeholder="Paste the full text here…" value={text} onChange={(e) => setText(e.target.value)} rows={12} />
        <div className="modal-actions">
          <button className="btn ghost" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={text.trim().length < 50} onClick={() => onSubmit(title, text)}>Add</button>
        </div>
      </div>
    </div>
  );
}
