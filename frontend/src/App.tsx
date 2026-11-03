import { useState } from "react";
import { Chat } from "./components/Chat";
import { FitPanel } from "./components/FitPanel";
import { Sidebar } from "./components/Sidebar";
import { useSession } from "./useSession";

export default function App() {
  const { state, setState, error, reset } = useSession();
  const [tab, setTab] = useState<"chat" | "fit">("chat");

  if (error) return <div className="center"><div className="notice error">Cannot reach the server: {error}</div></div>;
  if (!state) return <div className="center muted">Loading…</div>;

  const ready = !!state.resume && state.jobs.length > 0;
  return (
    <div className="app">
      <Sidebar state={state} onChange={setState} onReset={reset} />
      <main className="main">
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === "chat"} className={tab === "chat" ? "on" : ""} onClick={() => setTab("chat")}>Chat</button>
          <button role="tab" aria-selected={tab === "fit"} className={tab === "fit" ? "on" : ""} onClick={() => setTab("fit")}>Fit reports</button>
        </nav>
        {/* Chat stays mounted so switching tabs doesn't lose the conversation. */}
        <div hidden={tab !== "chat"} className="pane"><Chat key={state.session_id} state={state} ready={ready} /></div>
        <div hidden={tab !== "fit"} className="pane scroll"><FitPanel key={`${state.session_id}:${state.resume?.id ?? ""}`} sid={state.session_id} jobs={state.jobs} ready={ready} llm={state.llm_enabled} /></div>
      </main>
    </div>
  );
}
