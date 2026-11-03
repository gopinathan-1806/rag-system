export interface DocumentInfo {
  id: string;
  kind: "resume" | "job";
  label: string;
  title: string;
  filename: string;
  chunks: number;
  chars: number;
  warnings: string[];
}

export interface SessionState {
  session_id: string;
  resume: DocumentInfo | null;
  jobs: DocumentInfo[];
  llm_enabled: boolean;
}

export interface Source {
  id: string;
  label: string;
  title: string;
  section: string;
  text: string;
  cosine: number;
}

export type ChatEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; answer?: string; cited: string[]; grounded: boolean; stats?: { retrieval_ms: number; total_ms: number } }
  | { type: "error"; message: string };

export interface FitReport {
  job_id: string;
  job_label: string;
  job_title: string;
  match_score: number;
  summary: string;
  matched_skills: { skill: string; evidence: string; verified: boolean }[];
  missing_skills: { skill: string; importance: "required" | "preferred"; suggestion: string }[];
  experience_alignment: string;
  interview_questions: { question: string; talking_point: string }[];
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  cited?: string[];
  streaming?: boolean;
  error?: boolean;
}
