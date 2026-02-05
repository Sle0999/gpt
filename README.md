# OpenAI Responses Manifold – v25
### Fix release: OpenAI Responses built-in image generation tool wiring

This release focuses on fixing the **"tool not found"** error when Open WebUI requests image generation through the OpenAI Responses API manifold.

## What changed in v25

- Added a new manifold file:
  - `openai_responses_manifold_gpt52_pro_v25.py`
- Bumped manifold metadata version from **24 → 25**.
- Fixed built-in image generation tool registration:
  - When WebUI features indicate image generation is enabled (`image_generation` or `image_gen`) **and** the chosen model supports the built-in image tool, the manifold now appends:
    - `{"type": "image_generation"}`

## Why this fixes the issue

Previously, the manifold handled web search feature wiring but did not append the OpenAI Responses **image_generation** tool in the request tool list. That mismatch could lead to runtime failures where image requests attempted to call a tool the API had not been told to enable, resulting in **tool not found** behavior.

## Notes

- This is a targeted compatibility fix and keeps existing behavior unchanged for non-image requests.
- Existing v24 file is left intact; v25 is provided as a new filename/version as requested.
