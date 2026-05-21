# ECG Digitizer — Web Application

AI-powered web platform for converting printed ECG images into precise multi-lead digital signals.

## Architecture

```
web/
├── frontend/          Next.js 15 + TypeScript + Tailwind CSS + Framer Motion
├── backend/           FastAPI + WebSocket + Async Pipeline
├── docker-compose.yml Docker deployment
└── .env.example       Environment configuration
```

## Quick Start

### 1. Backend

```bash
cd web/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend will be available at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### 2. Frontend

```bash
cd web/frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`

### 3. Docker (Optional)

```bash
cd web
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload ECG image |
| POST | `/api/process/{job_id}` | Start pipeline processing |
| GET | `/api/status/{job_id}` | Check processing status |
| GET | `/api/results/{job_id}` | Get extracted signals + metrics |
| GET | `/api/metrics/{job_id}` | Get metrics only |
| GET | `/api/download/{job_id}/{type}` | Download CSV/JSON/PNG |
| WS | `/ws/progress/{job_id}` | Real-time progress stream |

## Pipeline Integration

The backend uses placeholder pipeline methods in `services/ecg_pipeline.py`.
To integrate your real ML pipeline, replace the method bodies in:

- `stage0_marker_detection()` — Use `stage0_model.py`
- `stage1_grid_detection()` — Use `stage1_model.py`
- `stage2_signal_segmentation()` — Use `stage2_model.py`
- `extract_12_leads()` — Convert segmentation to lead signals
- `generate_metrics()` — Compute SNR/RMSE/MAE against ground truth

## Tech Stack

- **Frontend**: Next.js 15, React 19, Tailwind CSS 4, Framer Motion, Recharts, Lucide Icons
- **Backend**: FastAPI, Uvicorn, WebSockets, NumPy, Pandas, Pillow
- **Deployment**: Docker, Docker Compose
