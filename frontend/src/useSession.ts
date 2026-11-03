import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api";
import type { SessionState } from "./types";

const KEY = "career-copilot-session";

/** Owns the server session: restores it after a refresh, or creates a fresh one. */
export function useSession() {
  const [state, setState] = useState<SessionState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    const created = await api.createSession();
    sessionStorage.setItem(KEY, created.session_id);
    setState(created);
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const saved = sessionStorage.getItem(KEY);
        if (saved) {
          try {
            setState(await api.getSession(saved));
            return;
          } catch (e) {
            if (!(e instanceof ApiError && e.status === 404)) throw e;
          }
        }
        await start();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not reach the server.");
      }
    })();
  }, [start]);

  const refresh = useCallback(async () => {
    if (state) setState(await api.getSession(state.session_id));
  }, [state]);

  const reset = useCallback(async () => {
    if (state) await api.deleteSession(state.session_id).catch(() => undefined);
    await start();
  }, [state, start]);

  return { state, setState, error, refresh, reset };
}
