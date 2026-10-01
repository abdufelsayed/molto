"""Bounded model smoke execution, owned by inference."""

from omlx_contracts.runtime import RuntimeOperationError


async def smoke(engine, model_type):
    if model_type in {"llm", "vlm"}:
        output = await engine.generate(
            "Reply with OK.", max_tokens=1, temperature=0.0, top_p=1.0
        )
        if not isinstance(getattr(output, "text", None), str):
            raise RuntimeError("Smoke generation returned an invalid text response")
        return {"summary": "Generated one token and validated the text response"}
    if model_type == "embedding":
        output = await engine.embed(["oMLX health check"])
        vectors = getattr(output, "embeddings", None)
        if not vectors or not vectors[0]:
            raise RuntimeError("Embedding smoke returned no vector")
        return {"dimensions": len(vectors[0])}
    if model_type == "reranker":
        output = await engine.rerank("health check", ["health check", "other"], top_n=2)
        if (
            getattr(output, "scores", None) is None
            or len(getattr(output, "indices", [])) != 2
        ):
            raise RuntimeError("Reranker smoke returned an invalid ranking")
        return {"documents": 2}
    if model_type == "audio_tts":
        data = await engine.synthesize("oMLX health check", max_tokens=64)
        if not isinstance(data, bytes) or len(data) < 44:
            raise RuntimeError("Speech smoke returned invalid audio")
        return {"output_bytes": len(data)}
    if model_type == "image_generation":
        data = await engine.generate_image(
            prompt="a gray square", seed=0, width=256, height=256, steps=1
        )
        if not isinstance(data, bytes) or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Image smoke returned invalid PNG")
        return {"output_bytes": len(data)}
    raise RuntimeOperationError("unavailable", "No smoke probe for this model type")
