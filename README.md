# OpenAI Responses Manifold – v25 “SmartRoute + Image Tooling Fixes”
### Improvements, Additions & Changes Made on Top of the Original

**Built on top of:**  
https://github.com/jrkropp/open-webui-developer-toolkit/tree/main/functions/pipes/openai_responses_manifold  
*(original author: Justin Kropp)*  
…but enhanced with a broad set of new features, a redesigned architecture, improved routing logic, and major optimizations throughout.

This README lists **only the modifications** introduced in my v25 fork — not the original feature set.

---

# 🚀 Major Enhancements in v25


## 0. 🛠 Responses API Image Tool Guard (New in v25)
Fixes a regression where requests without an explicit `tools` field could be sent as `tools: []`, causing image generation tool-choice failures.

### **What changed**
- Tools are now normalized **only when the request already includes `tools`**.
- If normalization yields no usable tools, the `tools` field is removed instead of sending `[]`.
- If `tool_choice.type == "image_generation"`, the manifold now guarantees `{"type":"image_generation"}` is present in `tools` before sending.
- Applied consistently to both streaming and non-streaming Responses requests.
- Legacy `tool_choice` requests for function `generate_image` are normalized to native `image_generation` so image calls route to Responses built-in tooling.
- The manifold now requests `include: ["image_generation_call.result"]` whenever image generation is enabled, then extracts image URLs/base64 from Responses output and appends renderable Markdown image blocks so generated images display in chat instead of only status text.

---

## 1. 🔮 SmartRoute System for `gpt-5-auto`
A full routing engine replaces the original static model mapping.

### **New routing capabilities**
- Uses a lightweight model (default: `gpt-4.1-nano`) to classify the prompt.
- Applies layered heuristics:
  - **Simple / short** → `gpt-4.1-nano`
  - **General / coding / mixed** → `gpt-4o`
  - **Reasoning-heavy** → `gpt-5.2` with appropriate effort
  - **Deep research / advanced math / physics** → `gpt-5.2-pro`
- Supports routing for your pseudo-model ecosystem.

### **New router valves**
- `GPT5_AUTO_ROUTER_MODEL`  
- `GPT5_AUTO_ROUTER_DEBUG`

### **New router debug output**
Shows:
- Route chosen by small model  
- Route chosen by fallback heuristic  
- Router disabled  
- Router failure → heuristic fallback

---

## 2. 🧱 Expanded Model & Pseudo-Model Support
v24 adds broad support for OpenAI’s newest models and your custom IDs.

### **New models supported**
- GPT-5.2 and GPT-5.2-Pro (forced `effort="high"`)  
- GPT-5.2 thinking tiers: low / medium / high / xhigh  
- Additional 4.1-nano / mini routing options  
- Full compatibility with o-series reasoning modes

### **New pseudo-model mapping layer**
Maps IDs like:
- `gpt-5-thinking-low/medium/high/xhigh`
- `gpt-5.2-thinking-*`
- `*-deep`
- `gpt-5.2-pro`
To:
- A real model  
- A reasoning effort level  
- A user-facing readable identity

---

## 3. 🪪 Identity Preamble (New)
The original manifold passed system instructions directly.

v24 injects a clean **identity header** so:
- The model reports itself using the **exact WebUI model ID you selected**, not an internal backend model.
- Meta-questions like *“what model are you?”* are answered correctly.

This applies to all pseudo-models and routed outputs.

---

## 4. 💰 Full Cost Tracking Engine (New)
The original manifold had **no cost accounting**.  
v24 introduces a full pricing system.

### **New features**
- Approximate pricing tables for:
  - GPT-5.2, 5.2-Pro  
  - GPT-4.1 (all tiers)  
  - GPT-4o  
  - Router model pricing  
  - `gpt-image-1` image pricing

- Tracks:
  - **Per-reply cost**
  - **Conversation total**
  - **Token + image cost**
  - **Cost from hidden image calls** (WebUI sometimes suppresses tool output)

### **New valves**
- `SHOW_COSTS`  
- `INCLUDE_IMAGE_COSTS`  
- `INLINE_COSTS_IN_MESSAGE`

### **New deduplication system**
Prevents duplicated or repeated cost lines when:
- A message regenerates  
- A turn errors and retries  
- WebUI sends multiple deltas

---

## 5. 🖼 Smarter Image Handling (New)
Major improvements over the original implementation.

### **New behavior**
- Detects image generation even when WebUI hides the tool call.
- Correctly estimates image count for cost pricing.
- Adds image model reference:
  - Example: `approx cost 1 image (gpt-image-1): $0.04`

---

## 6. 🔧 Input / Output Cleanup & Reliability Fixes
v24 removes multiple sources of 400-errors and malformed requests.

### **New sanitization**
- Strips unsupported WebUI fields  
- Normalizes model IDs  
- Ensures valid Responses API block structure  
- Fixes broken or empty system prompts  
- Ensures instructions + identity merge cleanly  
- Stabilizes multi-turn conversations involving hidden items

---

## 7. 🧠 Reasoning Effort & Summaries (Improved)
v24 modifies and extends the reasoning system.

### **Changes**
- Pseudo-models now map to correct `effort` levels.
- Router sets `effort` dynamically based on prompt.
- Optional reasoning summaries with `<details>` folding.
- Toggleable persistence of encrypted reasoning state.

---

## 8. 🔍 Behavior Improvements & Utilities
### **New enhancements**
- Better deduplication of hidden items  
- Improved rehydration of tool + image state across turns  
- Optional verbosity adjustment based on user instructions  
- Identity + reasoning preamble merges with system instructions safely

---

## 9. 🏷 Better Model Display Names (New in v25)
Model names shown in WebUI now expose explicit thinking level for key aliases.

### **Display behavior updates**
- `gpt-5-thinking` is shown as `OpenAI: gpt-5-thinking-medium` for explicit clarity.
- `gpt-5.2-thinking` is shown as `OpenAI: gpt-5.2-thinking-medium` for explicit clarity.
- `gpt-5.2` (and `gpt-5.2-chat-latest`) now display explicit `thinking: none`.
- IDs are unchanged; only the human-readable `name` is enhanced.

---

# 🧾 Summary of v25 Additions

v24 introduces **all of the following**, none of which exist in the original:

- ✔ SmartRoute engine for `gpt-5-auto`  
- ✔ Router-model classification + heuristic fallbacks  
- ✔ Expanded model support (5.2, 5.2-Pro, nano, mini, o-series)  
- ✔ Large pseudo-model mapping system  
- ✔ Identity preamble for correct “what model are you?” answers  
- ✔ Full cost accounting (tokens + images)  
- ✔ Image-cost inference when WebUI hides tool calls  
- ✔ New cost valves & inline/ toast behavior  
- ✔ Deduplication + retry protection  
- ✔ Improved reasoning effort system  
- ✔ Reasoning summaries / persistence controls  
- ✔ Request sanitization + schema fixes  
- ✔ More stable multi-turn hidden-item handling  
- ✔ Cleaner system + instruction merging  
- ✔ Better error safety & reliability

---

# 💬 Notes

This README represents **only the improvements** made in my fork (v25).  
All other architectural and base functionality belong to the original author.
