"""The JSON contract between the model steps and Python (pydantic v2)."""
from pydantic import BaseModel, Field


class NodeIn(BaseModel):
    title: str = Field(max_length=80)
    kind: str
    summary: str = Field(max_length=600)
    quotes: list[str] = Field(min_length=1, max_length=3)
    importance: int = Field(ge=1, le=5)


class EdgeIn(BaseModel):
    source: str
    target: str
    type: str
    quote: str | None = None


class Extraction(BaseModel):
    nodes: list[NodeIn]
    edges: list[EdgeIn] = []


class Classification(BaseModel):
    template: str
    reason: str


class Links(BaseModel):
    edges: list[EdgeIn]
    dropped: int = 0        # set by link(): edges removed for an unknown id, a disallowed type or a self-loop


class Verdict(BaseModel):
    id: str
    supported: bool
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class Revision(BaseModel):
    id: str
    summary: str | None          # None: the quotes do not support any summary of this node


class Revisions(BaseModel):
    revisions: list[Revision]
