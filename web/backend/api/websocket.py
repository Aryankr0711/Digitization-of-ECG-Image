"""
WebSocket handler for real-time pipeline progress.
"""
import asyncio
import json
import time
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set


class ConnectionManager:
    """Manages WebSocket connections per job_id."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self._connections:
            self._connections[job_id] = set()
        self._connections[job_id].add(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self._connections:
            self._connections[job_id].discard(websocket)
            if not self._connections[job_id]:
                del self._connections[job_id]

    async def send_progress(
        self,
        job_id: str,
        stage: str,
        stage_index: int,
        total_stages: int,
        progress: float,
        message: str,
    ):
        """Broadcast a progress event to all connections watching this job."""
        if job_id not in self._connections:
            return

        payload = json.dumps({
            "type": "progress",
            "job_id": job_id,
            "stage": stage,
            "stage_index": stage_index,
            "total_stages": total_stages,
            "progress": progress,
            "message": message,
            "timestamp": time.time(),
        })

        dead = set()
        for ws in self._connections[job_id]:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self._connections[job_id].discard(ws)

    async def send_complete(self, job_id: str, results_url: str):
        """Notify all watchers that processing is finished."""
        payload = json.dumps({
            "type": "complete",
            "job_id": job_id,
            "results_url": results_url,
            "timestamp": time.time(),
        })

        if job_id in self._connections:
            dead = set()
            for ws in self._connections[job_id]:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                self._connections[job_id].discard(ws)

    async def send_error(self, job_id: str, error: str):
        """Notify watchers of a pipeline error."""
        payload = json.dumps({
            "type": "error",
            "job_id": job_id,
            "error": error,
            "timestamp": time.time(),
        })

        if job_id in self._connections:
            for ws in list(self._connections[job_id]):
                try:
                    await ws.send_text(payload)
                except Exception:
                    pass


manager = ConnectionManager()
