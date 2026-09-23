from typing import Optional

from pydantic import BaseModel, Field, field_validator

MAX_QUERY_LEN = 300


class MapRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    k: int = Field(default=5, ge=1, le=10)
    include_experimental_reranker: bool = False

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    head_index_size: Optional[int] = None
    frozen_config_version: Optional[str] = None
    error: Optional[str] = None
