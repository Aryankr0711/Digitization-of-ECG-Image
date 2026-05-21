"""
REST API routes for ECG digitization.
"""
import uuid
import shutil
import json
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from core.config import settings
from models.schemas import UploadResponse, JobStatus, ProcessingStage
from services.ecg_pipeline import pipeline_service
from api.websocket import manager

router = APIRouter(prefix="/api")

# In-memory job store (swap for Redis/DB in production)
_jobs: dict[str, dict] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_ecg_image(file: UploadFile = File(...)):
    """Upload an ECG image and get a job_id for processing."""
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use PNG, JPG, or JPEG.")

    # Validate size
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large. Max {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB.")

    # Save file
    job_id = str(uuid.uuid4())[:8]
    job_dir = settings.UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / f"input{ext}"
    with open(image_path, "wb") as f:
        f.write(contents)

    _jobs[job_id] = {
        "status": ProcessingStage.UPLOADING,
        "progress": 0.0,
        "image_path": str(image_path),
        "output_dir": str(settings.OUTPUT_DIR / job_id),
        "results": None,
        "error": None,
    }

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        message="Upload successful. Call POST /api/process/{job_id} to start.",
    )


@router.post("/process/{job_id}")
async def start_processing(job_id: str, background_tasks: BackgroundTasks):
    """Start the ECG digitization pipeline for an uploaded image."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    job = _jobs[job_id]
    if job["results"] is not None:
        return {"message": "Already processed", "results_url": f"/api/results/{job_id}"}

    # Run pipeline in background
    background_tasks.add_task(_run_pipeline, job_id)
    return {"message": "Processing started", "job_id": job_id}


async def _run_pipeline(job_id: str):
    """Background task that executes the ECG pipeline."""
    job = _jobs[job_id]
    image_path = Path(job["image_path"])
    output_dir = Path(job["output_dir"])

    async def progress_callback(stage, stage_idx, total, progress, message):
        job["status"] = stage
        job["progress"] = progress
        await manager.send_progress(job_id, stage, stage_idx, total, progress, message)

    try:
        results = await pipeline_service.run_pipeline(
            image_path=image_path,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
        job["results"] = results
        job["status"] = ProcessingStage.COMPLETE
        job["progress"] = 1.0

        # Copy the original image to outputs for serving
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_dir / "original.png")

        await manager.send_complete(job_id, f"/api/results/{job_id}")

    except Exception as e:
        job["status"] = ProcessingStage.FAILED
        job["error"] = str(e)
        await manager.send_error(job_id, str(e))


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get current processing status for a job."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    job = _jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_stage=job["status"] if isinstance(job["status"], str) else job["status"].value,
        error=job.get("error"),
    )


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get the full results for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    job = _jobs[job_id]
    if job["results"] is None:
        raise HTTPException(202, "Processing not yet complete")

    results = job["results"]
    return {
        "job_id": job_id,
        "leads": [
            {"name": name, "data": data, "sample_rate": 500.0}
            for name, data in results["leads"].items()
        ],
        "lead_metrics": results["metrics"]["lead_metrics"],
        "average_metrics": results["metrics"]["average_metrics"],
        "processing_time_ms": results["processing_time_ms"],
        "original_image_url": f"/api/download/{job_id}/image",
        "stages_completed": results["stages_completed"],
    }


@router.get("/metrics/{job_id}")
async def get_metrics(job_id: str):
    """Get only the metrics for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    job = _jobs[job_id]
    if job["results"] is None:
        raise HTTPException(202, "Processing not yet complete")

    return job["results"]["metrics"]


@router.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download result files: 'csv', 'json', 'image', or 'png'."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    output_dir = settings.OUTPUT_DIR / job_id

    if file_type == "image":
        path = output_dir / "original.png"
        if path.exists():
            return FileResponse(path, media_type="image/png", filename=f"ecg_{job_id}_original.png")

    elif file_type == "json":
        path = output_dir / "metrics.json"
        if path.exists():
            return FileResponse(path, media_type="application/json", filename=f"ecg_{job_id}_metrics.json")

    elif file_type == "csv":
        # Generate CSV from leads.json
        leads_path = output_dir / "leads.json"
        csv_path = output_dir / "signals.csv"
        if leads_path.exists() and not csv_path.exists():
            import pandas as pd
            with open(leads_path) as f:
                leads = json.load(f)
            # Find max length
            max_len = max(len(v) for v in leads.values())
            data = {}
            for name, values in leads.items():
                padded = values + [0.0] * (max_len - len(values))
                data[name] = padded
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
        if csv_path.exists():
            return FileResponse(csv_path, media_type="text/csv", filename=f"ecg_{job_id}_signals.csv")

    elif file_type == "png":
        # Return a placeholder — in real pipeline this would be the rendered plot
        path = output_dir / "original.png"
        if path.exists():
            return FileResponse(path, media_type="image/png", filename=f"ecg_{job_id}_visualization.png")

    raise HTTPException(404, f"File type '{file_type}' not found for job {job_id}")
