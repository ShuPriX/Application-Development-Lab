import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, useCallback, useRef, useEffect } from "react";
import { getAnalysis, triggerAnalysis, createAnalysisWebSocket } from "../services/api";
import type { AnalysisProgress } from "../types/analysis";

export function useAnalysis(tmdbId: number | undefined) {
  return useQuery({
    queryKey: ["analysis", tmdbId],
    queryFn: () => getAnalysis(tmdbId!),
    enabled: !!tmdbId,
    staleTime: 30 * 60 * 1000,
  });
}

export function useAnalysisMutation(tmdbId: number | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => triggerAnalysis(tmdbId!),
    onSuccess: (data) => {
      queryClient.setQueryData(["analysis", tmdbId], data);
    },
  });
}

export interface AnalysisLiveState {
  latest: AnalysisProgress | null;
  logs: AnalysisProgress[];
}

export function useAnalysisProgress(tmdbId: number | undefined, active: boolean): AnalysisLiveState {
  const [state, setState] = useState<AnalysisLiveState>({ latest: null, logs: [] });
  const wsRef = useRef<WebSocket | null>(null);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!active || !tmdbId) {
      disconnect();
      setState({ latest: null, logs: [] });
      return;
    }

    const ws = createAnalysisWebSocket(tmdbId);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as AnalysisProgress;
      setState((prev) => ({
        latest: data,
        logs: [...prev.logs, data],
      }));
    };

    ws.onclose = () => {
      // keep state so the UI can show final results briefly
    };

    return disconnect;
  }, [tmdbId, active, disconnect]);

  return state;
}
