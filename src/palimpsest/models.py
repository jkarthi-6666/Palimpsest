from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, Field


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Event(BaseModel):
    # append_event replaces this with the content-derived ID. It is optional so a
    # producer does not have to invent an ID that the database layer will ignore.
    id: UUID | None = None
    user_id: UUID
    session_id: UUID
    actor: str
    text: str
    occurred_at: AwareDatetime
    ingested_at: AwareDatetime = Field(default_factory=lambda: datetime.now().astimezone())
    tombstoned: bool = False


class Memory(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    mem_type: MemoryType
    user_id: UUID
    agent_id: UUID | None = None
    session_id: UUID | None = None
    org_id: UUID | None = None
    actor: str
    confidence: float = Field(ge=0.0, le=1.0)

    # These remain separate because valid time describes when a fact is true,
    # while observed time describes when Palimpsest learned it.
    # Phase 1 leaves valid time unknown when the conversation does not state it.
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    observed_at: AwareDatetime

    superseded_by: UUID | None = None
    supersede_kind: str | None = None
    mention_count: int = Field(default=1, ge=0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: AwareDatetime | None = None
    entities: list[str] = Field(default_factory=list)
    sensitivity: str = "normal"
    source_event_ids: list[UUID] = Field(default_factory=list)
    extractor_version: str | None = None
    deleted_at: AwareDatetime | None = None
