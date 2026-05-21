"use client";
import { useEffect, useRef, useCallback, useState } from "react";
import type { ProcessingEvent } from "@/types/ecg";

interface UseWebSocketOptions {
  url: string | null;
  onMessage?: (event: ProcessingEvent) => void;
  onComplete?: (resultsUrl: string) => void;
  onError?: (error: string) => void;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

export function useWebSocket({
  url,
  onMessage,
  onComplete,
  onError,
  reconnectAttempts = 5,
  reconnectDelay = 2000,
}: UseWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const attemptsRef = useRef(0);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const connect = useCallback(() => {
    if (!url) return;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        attemptsRef.current = 0;
        // Ping every 15s to keep alive
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, 15000);
      };

      ws.onmessage = (event) => {
        try {
          const data: ProcessingEvent = JSON.parse(event.data);
          if (data.type === "pong") return;
          if (data.type === "complete" && data.results_url) {
            onComplete?.(data.results_url);
          } else if (data.type === "error" && data.error) {
            onError?.(data.error);
          } else {
            onMessage?.(data);
          }
        } catch {
          // Skip non-JSON messages
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
        // Reconnect with exponential backoff
        if (attemptsRef.current < reconnectAttempts) {
          const delay = reconnectDelay * Math.pow(2, attemptsRef.current);
          attemptsRef.current += 1;
          setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      // Connection failed
    }
  }, [url, onMessage, onComplete, onError, reconnectAttempts, reconnectDelay]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
    };
  }, [connect]);

  return { isConnected };
}
