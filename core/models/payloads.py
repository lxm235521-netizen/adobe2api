from __future__ import annotations

import time
from typing import Optional


def size_from_ratio(ratio: str, output_resolution: str = "2K") -> dict:
    level = (output_resolution or "2K").upper()
    if level == "1K":
        ratio_map = {
            "1:1": {"width": 1024, "height": 1024},
            "16:9": {"width": 1360, "height": 768},
            "9:16": {"width": 768, "height": 1360},
            "4:3": {"width": 1152, "height": 864},
            "3:4": {"width": 864, "height": 1152},
        }
    elif level == "4K":
        ratio_map = {
            "1:1": {"width": 4096, "height": 4096},
            "16:9": {"width": 5504, "height": 3072},
            "9:16": {"width": 3072, "height": 5504},
            "4:3": {"width": 4096, "height": 3072},
            "3:4": {"width": 3072, "height": 4096},
        }
    else:
        ratio_map = {
            "1:1": {"width": 2048, "height": 2048},
            "16:9": {"width": 2752, "height": 1536},
            "9:16": {"width": 1536, "height": 2752},
            "4:3": {"width": 2048, "height": 1536},
            "3:4": {"width": 1536, "height": 2048},
        }
    return ratio_map.get(ratio, ratio_map["16:9"])


def build_image_payload_candidates(
    *,
    prompt: str,
    aspect_ratio: str,
    output_resolution: str,
    upstream_model_id: str,
    upstream_model_version: str,
    source_image_ids: Optional[list[str]] = None,
) -> list[dict]:
    seed = int(time.time()) % 999999
    base_payload = {
        "modelId": upstream_model_id,
        "modelVersion": upstream_model_version,
        "n": 1,
        "prompt": prompt,
        "size": size_from_ratio(aspect_ratio, output_resolution),
        "seeds": [seed],
        "groundSearch": False,
        "skipCai": False,
        "output": {"storeInputs": True},
        "generationMetadata": {"module": "text2image"},
        "modelSpecificPayload": {
            "aspectRatio": aspect_ratio,
            "parameters": {"addWatermark": False},
        },
    }

    if not source_image_ids:
        # Upstream validation for nano-banana can vary across deploys.
        # Keep multiple compatible text2image payloads and let submit loop fallback.
        c1 = dict(base_payload)
        c1["referenceBlobs"] = []

        c2 = dict(base_payload)
        c2["referenceBlobs"] = []
        c2["generationMetadata"] = {
            "module": "text2image",
            "submodule": "ff-image-generate",
        }

        c3 = dict(base_payload)
        c3["referenceBlobs"] = []
        c3["generationMetadata"] = {
            "module": "text2image",
            "submodule": "ff-image-generate",
        }
        c3["modelSpecificPayload"] = {"parameters": {"addWatermark": False}}
        c3.pop("skipCai", None)

        return [c1, c2, c3]

    candidates: list[dict] = []
    edited = dict(base_payload)
    edited["generationMetadata"] = {"module": "image2image"}

    c1 = dict(edited)
    c1["referenceBlobs"] = [
        {"id": img_id, "usage": "general"} for img_id in source_image_ids
    ]
    candidates.append(c1)

    c4 = dict(edited)
    c4["referenceBlobs"] = []
    c4["imagePrompt"] = {"referenceImage": source_image_ids[0]}
    candidates.append(c4)

    c5 = dict(edited)
    c5["referenceBlobs"] = []
    c5["imagePrompt"] = {"referenceImage": {"id": source_image_ids[0]}}
    candidates.append(c5)

    c6 = dict(edited)
    c6["referenceBlobs"] = [
        {"id": img_id, "usage": "general"} for img_id in source_image_ids
    ]
    c6["generationMetadata"] = {
        "module": "image2image",
        "submodule": "ff-image-generate",
    }
    candidates.append(c6)

    return candidates
