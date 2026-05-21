"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ProcessingStage(str, Enum):
    UPLOADING = "uploading"
    STAGE_0 = "stage_0_marker_detection"
    STAGE_1 = "stage_1_grid_detection"
    STAGE_2 = "stage_2_signal_segmentation"
    EXTRACTING = "extracting_leads"
    METRICS = "generating_metrics"
    COMPLETE = "complete"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    message: str


class LeadData(BaseModel):
    name: str
    data: list[float]
    sample_rate: float = 500.0


class Metrics(BaseModel):
    snr_db: float
    rmse: float
    mae: float
    mse: float


class LeadMetrics(BaseModel):
    lead_name: str
    metrics: Metrics


class ECGResults(BaseModel):
    job_id: str
    leads: list[LeadData]
    lead_metrics: list[LeadMetrics]
    average_metrics: Metrics
    processing_time_ms: float
    original_image_url: str
    stages_completed: list[str]


class ProcessingEvent(BaseModel):
    """WebSocket message sent during pipeline execution."""
    job_id: str
    stage: str
    stage_index: int
    total_stages: int
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: Optional[float] = None


class JobStatus(BaseModel):
    job_id: str
    status: ProcessingStage
    progress: float
    current_stage: Optional[str] = None
    error: Optional[str] = None
