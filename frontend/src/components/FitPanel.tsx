import { useState } from "react";
import { api } from "../api";
import type { DocumentInfo, FitReport } from "../types";

interface Props {
  sid: string;
  jobs: DocumentInfo[];
  ready: boolean;
  llm: boolean;
}

export function FitPanel({ sid, jobs, ready, llm }: Props) {
  const [reports, setReports] = useState<Record<string, FitReport>>({});
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  async function generate(jobId: string, refresh = false) {
    setLoading(jobId);
    setError(null);
    setSelected(jobId);
    try {
      const r = await api.fit(sid, jobId, refresh);
      setReports((p) => ({ ...p, [jobId]: r }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not generate the report.");
    } finally {
      setLoading(null);
    }
  }

  const current = selected && jobs.some((j) => j.id === selected) ? reports[selected] : undefined;

  if (!ready || jobs.length === 0)
    return <div className="empty"><h2>Fit reports</h2><p>Add your resume and at least one job posting to generate a scored fit report.</p></div>;

  return (
    <div className="fit">
      <div className="fit-jobs">
        {jobs.map((j) => (
          <button key={j.id} className={`fit-job ${selected === j.id ? "sel" : ""}`} onClick={() => (reports[j.id] ? setSelected(j.id) : generate(j.id))} disabled={!llm || loading !== null}>
            <span className="tag job">{j.label}</span>
            <span className="fit-job-title">{j.title}</span>
            {reports[j.id] ? <ScoreBadge score={reports[j.id].match_score} /> : <span className="muted">{loading === j.id ? "Analyzing…" : "Analyze"}</span>}
          </button>
        ))}
      </div>
      {error && <div className="notice error" role="alert">{error}</div>}
      {loading && !current && <div className="notice busy" role="status">Reading your resume against the posting…</div>}
      {current && <Report r={current} onRefresh={() => generate(current.job_id, true)} busy={loading !== null} />}
      {!current && !loading && !error && <p className="muted pad">Select a job to generate a structured assessment.</p>}
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const tone = score >= 75 ? "good" : score >= 50 ? "mid" : "low";
  return <span className={`score ${tone}`}>{score}</span>;
}

function Report({ r, onRefresh, busy }: { r: FitReport; onRefresh: () => void; busy: boolean }) {
  const tone = r.match_score >= 75 ? "good" : r.match_score >= 50 ? "mid" : "low";
  return (
    <div className="report">
      <div className="report-head">
        <div className={`ring ${tone}`} style={{ ["--p" as string]: `${r.match_score * 3.6}deg` }}><span>{r.match_score}</span></div>
        <div>
          <h3>{r.job_label} — {r.job_title}</h3>
          <p>{r.summary}</p>
        </div>
        <button className="btn ghost" onClick={onRefresh} disabled={busy}>Regenerate</button>
      </div>

      <h4>Where you match</h4>
      <ul className="skills">
        {r.matched_skills.map((m) => (
          <li key={m.skill}>
            <span className="skill ok">{m.skill}</span>
            <q>{m.evidence}</q>
            {!m.verified && <em className="unverified" title="This quote could not be found verbatim in your resume">unverified</em>}
          </li>
        ))}
      </ul>

      <h4>Gaps to close</h4>
      <ul className="skills">
        {r.missing_skills.map((m) => (
          <li key={m.skill}>
            <span className={`skill ${m.importance === "required" ? "req" : "pref"}`}>{m.skill}</span>
            <small>{m.importance}</small>
            <p>{m.suggestion}</p>
          </li>
        ))}
        {r.missing_skills.length === 0 && <li className="muted">No notable gaps found.</li>}
      </ul>

      <h4>Experience alignment</h4>
      <p>{r.experience_alignment}</p>

      <h4>Interview prep</h4>
      <ol className="qa">
        {r.interview_questions.map((q) => (
          <li key={q.question}><b>{q.question}</b><p>{q.talking_point}</p></li>
        ))}
      </ol>
    </div>
  );
}
