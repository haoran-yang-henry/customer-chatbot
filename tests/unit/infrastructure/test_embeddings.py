import math

from customer_agent.infrastructure.embeddings import HashingEmbeddingProvider


async def test_hashing_embeddings_are_deterministic_and_normalised() -> None:
    provider = HashingEmbeddingProvider(dimensions=64)

    first, second = await provider.embed_documents(["Rücksendung", "Rücksendung"])

    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


async def test_similar_texts_are_closer_than_unrelated_texts() -> None:
    provider = HashingEmbeddingProvider(dimensions=256)
    query = await provider.embed_query("Versandkosten")
    similar, unrelated = await provider.embed_documents(["Versandkosten Österreich", "Akku"])

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, similar) > cosine(query, unrelated)
