from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ..., description="User question about internal knowledge base"
    )


class AskResponse(BaseModel):
    answer: str = Field(
        ..., description="Generated answer from the LLM"
    )
    citations: list[str] = Field(
        default_factory=list,
        description="List of supporting document references"
    )
    confidence: float = Field(
        0.0,
        description="Confidence score between 0 and 1"
    )
    latency_ms: float = Field(
        ..., description="End-to-end request latency in milliseconds"
    )
    model: str = Field(
        ..., description="Model name used for generation"
    )