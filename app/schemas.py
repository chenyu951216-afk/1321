from __future__ import annotations

from pydantic import BaseModel, Field


class FaceResponse(BaseModel):
    id: str
    name: str
    thumbnail_url: str
    created_at: str
    updated_at: str


class FaceListResponse(BaseModel):
    faces: list[FaceResponse]


class RenameFaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class DetectionFaceResponse(BaseModel):
    index: int
    bbox: list[int]
    thumbnail_url: str


class TargetDetectionResponse(BaseModel):
    token: str
    face_count: int
    faces: list[DetectionFaceResponse]


class SwapResponse(BaseModel):
    result_url: str
    download_url: str
    processing_time: float
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_status: str
    provider: str | None = None
    model_error: str | None = None

