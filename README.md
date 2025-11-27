# 🧠 OpenAI GPT 4 / 4o / 5 / 5.1 / 5-Pro Manifold for OpenWebUI

### Advanced Responses-API Router • Reasoning Engine • Image Support • Cost Tracking • Web Search • MCP • Tools

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenWebUI](https://img.shields.io/badge/OpenWebUI-Compatible-success)
![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-red)

A **fully custom, heavily-modified** OpenAI Responses-API manifold for OpenWebUI supporting:

✔ GPT-4
✔ GPT-4o (4o, 4o-mini, 4o-reasoning)
✔ GPT-5
✔ GPT-5.1
✔ GPT-5-Pro
✔ gpt-image-1
✔ o-series reasoning models
✔ MCP tools
✔ Web search preview
✔ Full cost accounting for tokens + images
✔ Pseudo models for high-effort reasoning

Built on top of:
[https://github.com/jrkropp/open-webui-developer-toolkit/tree/main/functions/pipes/openai_responses_manifold](https://github.com/jrkropp/open-webui-developer-toolkit/tree/main/functions/pipes/openai_responses_manifold)
…but **heavily expanded**, re-architected, optimized, and enhanced.

---

# 🚀 Features

Below is a **complete feature breakdown** of everything this manifold does.

---

# 1. OpenAI Responses API Bridge for OpenWebUI

* Converts WebUI-style requests → **OpenAI Responses API** format.
* Normalizes model names and strips `openai_responses.` prefix.
* Injects an identity preamble so when users ask *“what model are you?”*, the answer reflects the **WebUI-visible pseudo model**:

  * Example: `gpt-5-thinking-high`, `gpt-5-auto`, etc.

---

# 2. Model Routing, Pseudo-Models & Reasoning Effort

Supports an extensive set of **pseudo models** that map to real OpenAI models with reasoning settings:

| Pseudo Model          | Actual Model | Effort |
| --------------------- | ------------ | ------ |
| gpt-5-thinking        | gpt-5        | medium |
| gpt-5-thinking-high   | gpt-5        | high   |
| gpt-5.1-thinking-high | gpt-5.1      | high   |
| o3-mini-high          | o3-mini      | high   |
| o4-mini-high          | o4-mini      | high   |

### gpt-5-auto Router

* Automatically chooses a best-fit model.
* Applies reasoning levels based on your rules.

### GPT-5-Pro Special Handling

* Forced `effort="high"`
* Non-streaming only (due to API restrictions)

---

# 3. Reasoning Features & Summaries

### Supported Features

* `reasoning.effort`
* `reasoning.summary` (visible chain-of-thought summaries)

### Summary Valve

`REASONING_SUMMARY = auto | concise | detailed | disabled`

### Persisted Reasoning Tokens Valve

`PERSIST_REASONING_TOKENS = disabled | response | conversation`

Allows OpenAI to carry encrypted reasoning forward.

### UI Integration

* Uses `<details>` blocks to create:

  * **Thinking…**
  * reasoning summary
  * **Done thinking!**

Clean, collapsible, fully readable without polluting main output.

---

# 4. Tools, Web Search, and MCP

### Function Calling

* Converts WebUI tools → **strict** Responses-API JSON schemas.
* If native function calling is disabled in WebUI, the manifold:

  * Automatically patches the model config
  * Displays a toast message instructing user to retry

### Web Search (web_search_preview)

Enabled when:

* Model supports it (4.1, 4o, o-series)
* Valve on
* Effort ≠ minimal

Provides:

* Context size tuning
* Optional user location
* URL tracking + numbered citations
* Source panel events

### MCP Integration

Automatically loads all MCP servers defined in:

```
REMOTE_MCP_SERVERS_JSON
```

---

# 5. Image Support (Input + Generation)

### 5.1 Image Input (User → Model)

All WebUI content blocks are converted:

| WebUI      | Responses API |
| ---------- | ------------- |
| text       | input_text    |
| image_url  | input_image   |
| input_file | input_file    |

### 5.2 Image Generation (Model → User)

* Detects `image_generation_call`
* Displays: **“🎨 Let me create that image…”**
* Tracks generated images
* Estimates image count when OpenWebUI hides tool calls

---

# 6. Cost Estimation System (Tokens + Images)

### 6.1 Built-in Pricing

#### Token Pricing

Supports:

* GPT-5, GPT-5.1, GPT-5-Pro
* GPT-4.1, 4.1-mini, 4.1-nano
* GPT-4o text

#### Image Pricing

* `gpt-image-1` @ 1024×1024 → **$0.04 per image**

---

### 6.2 Per-Conversation Cost Tracking

* Local session DB maintains cumulative totals by `chat_id`
* Each assistant turn:

  * Summarizes cost
  * Updates cumulative totals
  * Displays inline or toast cost depending on settings

---

### 6.3 Cost Valves

| Valve                   | Behavior                                 |
| ----------------------- | ---------------------------------------- |
| SHOW_COSTS              | Enables or disables cost system entirely |
| INCLUDE_IMAGE_COSTS     | Token-only or token+image pricing        |
| INLINE_COSTS_IN_MESSAGE | Inline or toast display                  |

### Example Cost Output

```
[approx cost this reply (gpt-5-thinking-high → gpt-5): $0.00019 | approx total: $0.00019]
```

### Deduplication

Automatically removes old cost lines to prevent stacking.

---

### 6.4 Image Cost Inference

If WebUI hides image API calls, the manifold:

* Analyzes assistant text
* Infers image generation
* Applies default pricing

Example:
If output text says *“Here is your generated image”*, cost system infers `1 image`.

---

# 7. Message & History Handling

* Persists hidden items (tool calls, reasoning, images)
* Embeds invisible markers in messages
* Re-hydrates previous items into `input[]` for multi-turn continuity

---

# 8. Verbosity Control

Reactively adjusts output length:

| User Message   | Effect           |
| -------------- | ---------------- |
| “Add details”  | verbosity = high |
| “More concise” | verbosity = low  |

Automatically removes the trigger message and regenerates the response.

---

# 📦 Installation

### 1. Navigate to your OpenWebUI directory:

```
~/.config/open-webui/pipes
```

### 2. Add the manifold

Place your file here:

```
openai_responses_manifold_gpt51_pro_v21.py
```

### 3. Restart OpenWebUI

The pipeline will load automatically.

---

# ⚙️ Optional: OpenWebUI Model Config Template

Example model entry:

```json
{
  "name": "gpt-5-thinking-high",
  "id": "openai_responses.gpt-5-thinking-high",
  "provider": "openai_responses",
  "mode": "chat",
  "native_tools": true,
  "native_tool_calling": true
}
```

---

# 🧩 File Structure (Diagram)

```
openai_responses_manifold/
│
├── model_aliases/        # Pseudo model → real model logic
├── routers/              # gpt-5-auto selection logic
├── utils/
│   ├── markers.py        # Hidden item marker system
│   ├── costs.py          # Pricing + cost generation
│   ├── items.py          # Persistence + rehydration
│   └── messages.py       # Transformer for WebUI → Responses API
│
└── manifold.py           # Main pipeline implementation
```

*(Directory names for illustration — adapt based on your actual layout.)*

---

# 🧪 Sample Usage

Ask a deep reasoning question:

```
Explain Gödel’s incompleteness theorem in the style of a physics textbook.
```

Request an image:

```
Generate a cyberpunk cityscape at night with neon fog.
```

Enable high reasoning:

```
Use high reasoning effort for the next answer.
```

Ask to expand detail:

```
Add details
```

---

# 📝 License

MIT License – free to modify and redistribute.

---

# ❤️ Acknowledgements

* Original manifold by **@jrkropp**
* Extended & upgraded into a full multi-model, multi-system router
* Designed specifically for developers using **OpenWebUI + OpenAI Responses API**

---
