import os
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointIdsList, PointStruct, VectorParams


load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

_client: QdrantClient | None = None


def connect() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        if not url:
            raise RuntimeError("QDRANT_URL is not set; copy .env.example to .env")
        _client = QdrantClient(url=url)
    return _client


def ensure_collection(name: str, dim: int) -> None:
    client = connect()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        return

    existing_dim = client.get_collection(name).config.params.vectors.size
    if existing_dim != dim:
        raise ValueError(
            f"collection {name!r} has dimension {existing_dim}, expected {dim}"
        )


def upsert(point_id: UUID | str, vector: list[float], payload: dict[str, Any]) -> None:
    connect().upsert(
        collection_name="memories",
        points=[PointStruct(id=str(point_id), vector=vector, payload=payload)],
        wait=True,
    )


def search(vector: list[float], k: int, filter: Filter | None = None) -> list[Any]:
    result = connect().query_points(
        collection_name="memories",
        query=vector,
        limit=k,
        query_filter=filter,
    )
    return list(result.points)


def delete(point_id: UUID | str) -> None:
    connect().delete(
        collection_name="memories",
        points_selector=PointIdsList(points=[str(point_id)]),
        wait=True,
    )
