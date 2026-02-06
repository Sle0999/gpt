# OpenAI Responses Manifold – v34 “SmartRoute”
### Current Features + Full Repository Changelog

**Built on top of:**  
https://github.com/jrkropp/open-webui-developer-toolkit/tree/main/functions/pipes/openai_responses_manifold  
*(original author: Justin Kropp)*  
This fork expands the Responses API manifold with routing, model aliasing, tool normalization, cost reporting, and extra controls.

---

# ✅ Current Feature Set (v34)

## 1. 🔮 SmartRoute for `gpt-5-auto`
- Uses a lightweight router model to pick the best target for `gpt-5-auto`.  
- Adds configurable router valves:
  - `GPT5_AUTO_ROUTER_MODEL` (default: `gpt-4.1-nano`)  
  - `GPT5_AUTO_ROUTER_DEBUG` (optional debug suffix)  
【F:openai_responses_manifold_gpt52_pro_v34.py†L925-L939】

## 2. 🧱 Model Catalog + Pseudo-Model Aliases
- Default model list includes `gpt-5-auto`, `gpt-5.2`, `gpt-5.2-pro`, `gpt-5.2-chat-latest`, `gpt-5` thinking tiers, `gpt-4.1-nano`, `chatgpt-4o-latest`, `o3`, and `gpt-4o`.  
- Supports pseudo IDs for thinking tiers (low/medium/high/xhigh), plus mini/nano variants.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L907-L920】
- Alias map converts pseudo IDs to real models + reasoning effort levels (e.g., `gpt-5-thinking-high` → `gpt-5.2` + `high`).  
【F:openai_responses_manifold_gpt52_pro_v34.py†L358-L387】

## 3. 🪪 Identity Preamble
- Injects a lightweight identity header so the model reports the **exact WebUI model ID** selected by the user.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L833-L857】

## 4. 🧠 Reasoning Summaries + Persistence
- Optional reasoning summaries: `auto`, `concise`, `detailed`, or `disabled`.  
- Optional encrypted reasoning token persistence per response or per conversation.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L941-L960】

## 5. 🧰 Tooling + Execution Controls
- Parallel tool calls toggle, max tool-call limits, and max function-loop cycles.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L953-L975】
- Built-in OpenAI web search tool support with context sizing + user location configuration.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L977-L989】
- Optional persistence of tool results across turns.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L991-L995】
- Experimental remote MCP server auto-attach support.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L1024-L1035】

## 6. 🔧 Tool Normalization + Image Tool Mapping
- Normalizes function tools to the Responses API shape, deduplicates by name, and forces non-strict function tools.  
- Maps OpenWebUI image function tools (`generate_image`, `create_image`, `image_generation`) to OpenAI’s native `image_generation` tool.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L42-L139】
- Converts OpenWebUI message lists into Responses API `input` blocks during request sanitization.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L860-L868】
- Converts OpenWebUI image tool choices to native Responses tool selection during request sanitization.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L860-L883】

## 7. 💰 Cost Tracking + Image Cost Estimation
- Approximate pricing tables for GPT‑5.2/5.2‑Pro, GPT‑4.1, GPT‑4o, and `gpt-image-1`.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L157-L187】
- Optional per-response cost summaries with inline or toast-style output.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L997-L1017】
- Includes conservative image-cost fallback when metadata is unavailable.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L189-L193】
- Deduplicates cost summaries by chat + message to avoid repeated cost lines.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L196-L205】

## 8. 🧩 Reliability, Privacy, and Logging
- Truncation strategy control (`auto` or `disabled`) and service-tier selection (`auto`, `default`, `flex`, `priority`).  
【F:openai_responses_manifold_gpt52_pro_v34.py†L1037-L1052】
- Prompt cache key selection for privacy vs. cache efficiency (`id` or `email`).  
【F:openai_responses_manifold_gpt52_pro_v34.py†L1054-L1062】
- Configurable log level and optional marker display for debugging.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L1064-L1093】
- Redacts secrets from logs to avoid leaking API keys.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L30-L76】

---

# 🧾 Repository Changelog

## v34 “SmartRoute” (current)
- GPT‑5.2 model family + thinking tiers, updated alias mappings, and expanded pseudo‑model support.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L157-L187】【F:openai_responses_manifold_gpt52_pro_v34.py†L358-L387】
- SmartRoute engine and router valves for `gpt-5-auto`.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L925-L939】
- Tool normalization, image tool mapping, and secret‑safe logging.  
【F:openai_responses_manifold_gpt52_pro_v34.py†L30-L139】
- Note: version numbers v25–v33 were intentionally skipped in this repository’s numbering scheme.

## v23 “SmartRoute” (previous)
- Introduced `gpt-5-auto` routing valves and router debug output.  
【F:openai_responses_manifold_gpt51_pro_v23_smartroute.py†L873-L885】
- Added tool normalization + secret redaction for outbound logging.  
【F:openai_responses_manifold_gpt51_pro_v23_smartroute.py†L47-L122】

## v22
- Added conservative per‑image cost estimation when size/quality metadata is missing.  
【F:openai_responses_manifold_gpt51_pro_v22.py†L70-L74】

## v21 (baseline fork)
- Established cost tracking valves, identity preamble injection, and initial `gpt-5-auto` routing heuristics.  
【F:openai_responses_manifold_gpt51_pro_v21.py†L657-L681】【F:openai_responses_manifold_gpt51_pro_v21.py†L794-L809】【F:openai_responses_manifold_gpt51_pro_v21.py†L2271-L2280】
