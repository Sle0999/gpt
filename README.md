# OpenAI Responses API Manifold — v37.3

An Open WebUI function that connects conversations and tools to OpenAI's Responses API, with configurable Auto routing, reasoning controls, cost estimates, and a Frontier spending guard.

**Current source:** [openai_responses_manifold_v37.py](openai_responses_manifold_v37.py)

**Source version:** `37.3` · **Minimum Open WebUI version:** `0.11.0`

**Release date:** October 11, 2026 (Pacific Time)

**Built on top of:**  
https://github.com/jrkropp/open-webui-developer-toolkit/tree/main/functions/pipes/openai_responses_manifold  
*(original author: Justin Kropp)*  
This fork expands the Responses API manifold with routing, model aliasing, tool normalization, cost reporting, and extra controls.

## Install or upgrade

1. Use Open WebUI `0.11.0` or newer. Download the current Python source linked above, or copy its raw contents.
2. In Open WebUI's admin Functions area, add/import the source for a new installation, or replace the code of the existing v37 function for an upgrade. Keep the function ID `openai_responses_v37_canary` when updating an existing v37 installation.
3. Supply credentials privately through the server's `OPENAI_API_KEY` environment variable or the function's `API_KEY` Valve. The default API endpoint is `https://api.openai.com/v1`; `OPENAI_API_BASE_URL` or the `BASE_URL` Valve can configure another endpoint.
4. Save and enable the function, then review its saved Valves. Keep your intended tier assignments, Frontier mode, and budget limits. New source defaults **do not automatically overwrite existing saved Valve overrides**.
5. Check `MODEL_ID`, `AUTO_ROUTER_CLASSIFIER_MODEL`, all `AUTO_ROUTER_*_MODEL` assignments, and any saved model presets for the retiring `gpt-4.1-nano` or `gpt-5.4-nano`. Change a saved assignment if it still selects either model. The release defaults use the GPT-6 models listed below.
6. Verify a normal conversation with **OpenAI: Auto v2** and your intended direct model entries. Optional `AUTO_ROUTER_DEBUG` shows the selected tier, model, reasoning effort, confidence, and reason as a toast or below the response.

The source filename keeps `v37` for this patch release. The function ID remains `openai_responses_v37_canary`, and Auto's ID remains `openai-auto-v2-canary`. These identifiers preserve existing v37 chats and settings. Existing v36 functions use a separate ID and can coexist; replacing v36 with v37 requires reviewing your saved settings and choosing the v37 model entries.

Publish the Python source and documentation only. Keep API keys, `.env` files, saved Valve exports, chat exports, logs, database files, private endpoint settings, and the Frontier budget ledger outside the public repository. Prefer `INFO` or `WARNING` logging; `DEBUG` can include conversation and request content despite best-effort secret redaction.

## Current routing defaults

The default model list is `openai-auto-v2-canary, gpt-6-luna, gpt-6-sol, gpt-6-astra`. The function also exposes the concrete models assigned to its tiers, so changing a tier in Valves updates the available entries without editing Python.

| Role | Model default | Reasoning effort default |
| --- | --- | --- |
| Auto classifier | `gpt-6-luna` | `none` for the classifier request |
| Fast | `gpt-6-luna` | `none` |
| Balanced | `gpt-6-sol` | `low` |
| Deep | `gpt-6-sol` | `high` |
| Frontier | `gpt-6-astra` | `high` |
| Fallback | `gpt-6-sol` | No fixed router effort when routing is disabled |

Auto chooses a stable tier and then applies the model configured for that tier. Explicit tier/model requests are handled first; other requests use a structured-output classifier. A missing classifier, failed request, or invalid classification falls back to deterministic heuristics. The classifier receives a conversation excerpt and request metadata; its API call can add cost and latency.

`AUTO_ROUTER_POLICY` supports `economy`, `balanced` (default), and `best`. Economy favors Fast; Best favors Deep when justified. Classifier suggestions can adjust reasoning effort within the policy and `AUTO_ROUTER_MAX_REASONING_EFFORT` ceiling, which defaults to `xhigh`. Turning off `AUTO_ROUTER_ENABLED` sends Auto requests to the configured fallback model.

Legacy Auto IDs `gpt-5-auto` and `openai-auto` remain accepted by the active Auto v2 path for existing chats. The unused older router has been removed.

## Frontier controls

`FRONTIER_MODE` controls the configured Frontier model and protected model IDs:

| Mode | Behavior |
| --- | --- |
| `disabled` | Protected Frontier requests move to a configured lower tier. |
| `manual_only` (default) | Direct selection of the concrete Frontier model can use it after spending checks. Auto requests stay below Frontier, including prompts that name Frontier. |
| `automatic_under_budget` | Auto may choose Frontier after the mode and spending checks pass. Explicit Frontier requests inside Auto also require `AUTO_ROUTER_ALLOW_EXPLICIT_FRONTIER`. |

The configured Frontier model is always protected; `FRONTIER_PROTECTED_MODEL_IDS` can protect additional IDs. The classifier cannot use a protected Frontier model, and background title/tag tasks are moved away from Frontier.

Before a protected model call, the guard obtains an exact input-token count, calculates a conservative cost reservation from configured prices, limits output tokens, and checks a persistent SQLite ledger. It accounts for further model calls after tool execution within the same user turn. When usage is available, it reconciles the reservation using locally estimated usage cost.

| Guard setting | Source default |
| --- | --- |
| Maximum estimated cost per user turn | `$0.50` |
| Daily reservation budget | `$2.00` |
| Monthly reservation budget | `$20.00` |
| Frontier calls reached through Auto per day | `3` |
| Maximum automatically selected input tokens | `100,000` |
| Maximum output tokens, including reasoning | `8,000` |
| Minimum usable output budget before fallback | `512` tokens |
| Token-price safety multiplier | `1.10` |
| Additional reserve per call | `$0.05` |

The automatic call cap includes Auto prompts that explicitly request Frontier. Directly selecting a concrete Frontier model still requires the turn and period budget checks.

In Pacific Time, ledger midnight falls at **4:00 p.m. PST or 5:00 p.m. PDT on the preceding calendar date** (`00:00 UTC`). The ledger's daily and monthly accounting periods follow UTC boundaries.

Paid built-in tools are blocked for Frontier by default. Guarded calls use the standard service tier by default. Token-count or ledger failures fall back unless `FRONTIER_FAIL_OPEN_ON_PREFLIGHT_ERROR` is explicitly enabled. Keep `FRONTIER_PRICED_MODEL_ID` and the configured token rates aligned with the selected Frontier model; unmatched models or missing rates cause fallback.

Budget amounts and displayed costs are local estimates based on configured prices and available usage. Prices, caching, service tiers, and built-in tool charges can affect the final invoice. Review prices before enabling automatic Frontier use, and retain the ledger in persistent private storage.

## Other current features

- **Responses compatibility:** converts Chat Completions-style messages, structured-output settings, token limits, and reasoning parameters into Responses requests. Applies request sanitization to streaming and non-streaming paths, including classifier and background task calls.
- **GPT-6 reasoning controls:** maps legacy `minimal` effort to `low`; Astra's `none` also becomes `low`. Removes incompatible sampling and logprob parameters for GPT-6 reasoning requests.
- **Model aliases and identity:** supports retained pseudo-model aliases and adds an identity preamble. Tier models can be changed in Valves, with model dropdown discovery using environment-backed connection settings.
- **Tools and integrations:** normalizes function tools, deduplicates definitions, maps image-generation function names to the native image tool, and supports built-in web search and optional remote MCP servers. Custom tool execution has timeout, call-count, output-size, and loop limits.
- **Reasoning and persistence:** optional reasoning summaries and encrypted reasoning persistence; tool results can persist across conversation turns. Availability depends on model and account capabilities.
- **Cost display:** optional approximate per-response and conversation costs, including optional image estimates and classifier cost. Output can appear inline or as a toast.
- **Response handling:** supports streaming and non-streaming replies, refusal text, incomplete-response notices, and modern awaitable Open WebUI helpers.
- **Privacy and operation:** hashes the selected local account ID or email for prompt-cache and safety identifiers. Provides truncation, service-tier, logging, and internal-marker controls. Secret redaction is best effort; keep operational data private.

## Repository changelog

### v37.3 — current

- Publishes the v37.2 feature baseline as [openai_responses_manifold_v37.py](openai_responses_manifold_v37.py), requiring Open WebUI `0.11.0` or newer.
- Removes bundled references to `gpt-4.1-nano` and `gpt-5.4-nano`, plus the unused legacy router. Current routing uses the configured Auto v2 tier models.
- Keeps the existing v37 function and Auto identifiers, GPT-6 routing defaults, and Frontier spending guard.
- Updates installation, upgrade, configuration, privacy, and feature documentation. Existing saved Valve assignments still need review when upgrading.

### v37.2 — baseline included in v37.3

- Configurable Auto v2 routing across Fast, Balanced, Deep, and Frontier, with Luna/Sol/Astra defaults, a structured classifier, heuristic fallback, and reasoning-effort limits.
- Protected Frontier models with manual/automatic modes, input-token preflight, conservative turn/period reservations, and persistent budget accounting.
- Responses request compatibility updates, GPT-6 reasoning normalization, bounded custom tool execution, and hashed cache/safety identifiers.

### v36 — historical repository release

- [openai_responses_manifold_gpt54_family_v36.py](openai_responses_manifold_gpt54_family_v36.py) remains available as a historical release.
- Includes model-family and alias updates, plus handling for awaitable tool helpers returned by newer Open WebUI versions.
- Historical files retain the defaults and compatibility code of their release; use the current source for new installations.

### v34 “SmartRoute”

- GPT-5.2 model family and thinking tiers, updated alias mappings, and expanded pseudo-model support.
- SmartRoute engine and router Valves for `gpt-5-auto`.
- Tool normalization, image tool mapping, and secret-safe logging.
- Version numbers v24–v33 were intentionally skipped in this repository's numbering scheme.

### v23 “SmartRoute”

- Introduced `gpt-5-auto` routing Valves and router debug output.
- Added tool normalization and secret redaction for outbound logging.

### v22

- Added conservative per-image cost estimation when size/quality metadata is missing.

### v21 — baseline fork

- Established cost tracking Valves, identity preamble injection, and initial `gpt-5-auto` routing heuristics.
