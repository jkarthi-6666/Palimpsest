from uuid import UUID

from qdrant_client.models import FieldCondition, Filter, MatchValue

from palimpsest.db import get_memories_by_ids
from palimpsest.embeddings import embed
from palimpsest.models import Memory
from palimpsest.vectors import search


def retrieve(query: str, user_id: UUID, k: int = 15) -> list[Memory]:
    query_vector = embed([query], input_type="query")[0]
    query_filter = Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))]
    )
    hits = search(query_vector, k, query_filter)
    return get_memories_by_ids([UUID(str(hit.id)) for hit in hits])
