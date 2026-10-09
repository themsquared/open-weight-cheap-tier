# open-weight-cheap-tier

Which open-weight model should answer the cheap first call in a
[confidence-routed agent](https://github.com/themsquared/confidence-routed-agents)?
Two local candidates, served by Ollama, and Claude Sonnet 5 as the frontier
baseline, all three behind **one agentgateway v1.6.0 OpenAI-compatible route**,
scored on the same 60-case hand-labelled agent tool-call risk classification
set from [jev-benchmark](https://github.com/themsquared/jev-benchmark).

**Headline finding:** neither open-weight model is a drop-in cheap tier through
this interface. One of them failed to produce a parseable answer on **55% of
calls** (33/60). The other answered reliably but took **15-20x longer per call**
than Claude Sonnet 5, the model it was supposed to be the budget alternative to.

> 📖 **Read the write-up:** (added once live)

## Why these two models, not the four the field is discussing

HN this month is arguing about Qwen3.8 Flash Next (a 125B-parameter model run
on consumer hardware), Reflection's 501B open-weight release, and GLM-5.3. None
of that is what this repo tested. What was actually available on this machine
(Apple M4 Max, no CUDA) and already pulled in Ollama:

- **qwen3.8** — 27.3B parameters, Q4_K_M quantization, `qwen35` architecture
  family, 262K context. This is **not** the 125B "Flash Next" variant the HN
  thread is about; it is a different, smaller model that happens to share a
  name prefix. Said plainly here because conflating the two would misreport
  which model was actually measured.
- **nemotron-3.5-lightning** — 32.9B parameters, Q4_K_M, `nemotron_h_moe`
  architecture (sparse MoE), 1M context.

GLM-5.3 Flash and DeepSeek V4.1 Flash were not tested: no hosted API key for
either, and both are too large to run locally on this hardware without a
further quantization pass this run didn't attempt. Say so rather than silently
narrowing scope.

## Architecture

```
tasks.jsonl (60 cases, from jev-benchmark, Apache-2.0)
        |
        v
  bench.py --model <name>
        |
        v
  agentgateway v1.6.0 (docker-compose, port 3310)
   /v1/chat/completions
        |
   +----+----+----------------+
   |         |                |
qwen3.8  nemotron-3.5-     claude-sonnet-5
(ollama) lightning (ollama)  (anthropic)
   |         |                |
   +----Ollama :11434----+    Anthropic API
   (host.docker.internal)
```

One gateway, one route shape, three backends. The latency and token numbers
below all come from the gateway's own request/response cycle, not from three
different client libraries timing themselves.

## Quickstart

```bash
git clone https://github.com/themsquared/open-weight-cheap-tier
cd open-weight-cheap-tier
ollama pull qwen3.8
ollama pull nemotron-3.5-lightning
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up -d
python3 bench.py --model qwen3.8
python3 bench.py --model nemotron-3.5-lightning
python3 bench.py --model claude-sonnet-5
```

## The gotcha: the OpenAI-compatible path can't turn off thinking

Both Ollama models are reasoning models by default. Ollama's **native**
`/api/chat` endpoint honors `"think": false` and returns a clean answer in
~0.5s. Ollama's **OpenAI-compatible** `/v1/chat/completions` endpoint —
which is the only shape agentgateway's `provider: ollama` speaks — silently
ignores `"think": false` entirely, whether sent through the gateway or
straight to Ollama. Confirmed both ways: identical result, an empty
`content` field and the whole `max_tokens` budget spent on an
unterminated `reasoning` field.

There is no parameter fix on the OpenAI-compatible path. The only lever is
`max_tokens` large enough for the model to finish its reasoning block and
still emit the answer. This repo used 700 for qwen3.8 and 900 for
nemotron-3.5-lightning (Claude Sonnet 5 does not think unless asked, so it
keeps jev-benchmark's original 256) and **that was not enough**: qwen3.8 still
hit the cap on 6 of 60 calls, and nemotron-3.5-lightning on 33 of 60. Every
one of those calls returned HTTP 200 with a well-formed OpenAI response
envelope and an empty `content` field; nothing about the response shape
signals failure to a caller that isn't specifically checking for it.

## Results (n=60 each, 2026-10-09)

| | claude-sonnet-5 | qwen3.8 | nemotron-3.5-lightning |
|---|---|---|---|
| Answered | 59/60 (98.3%) | 54/60 (90.0%) | 27/60 (45.0%) |
| Accuracy of answered | 55/59 (93.2%) | 53/54 (98.1%) | 27/27 (100%) |
| **Accuracy of all 60** | **55/60 (91.7%)** | **53/60 (88.3%)** | **27/60 (45.0%)** |
| Latency p50 | 1,080ms | 18,885ms | 6,512ms |
| Latency p95 | 1,412ms | 42,368ms | 8,571ms |
| Avg input tokens | 240 | 161 | 161 |
| Avg output tokens | 22 | 316 | 644 |
| $/call | $0.000704 | $0 | $0 |
| Total for n calls | $0.0415 | $0 | $0 |

Accuracy by difficulty (of cases answered):

| | claude-sonnet-5 | qwen3.8 | nemotron-3.5-lightning |
|---|---|---|---|
| clear (34) | 34/34 (100%) | 34/34 (100%) | 21/21 (100%) |
| ambiguous (14) | 10/13 (76.9%) | 10/11 (90.9%) | 3/3 (100%) |
| adversarial (12) | 11/12 (91.7%) | 9/9 (100%) | 3/3 (100%) |

Every model that answered a case got the easy third right. The failures
split between wrong answers (Claude, mostly on the ambiguous slice) and no
answer at all (both Ollama models, worse on harder cases that needed more
reasoning tokens).

## What this does and does not show

- This is accuracy, malformed-response rate and latency on one 60-case
  classification task, not a general open-weight-vs-frontier verdict.
- "Cost" for the Ollama models is $0 because the hardware is already paid for
  and idle otherwise. That is real for a side project; it is not the same
  comparison as a cloud GPU instance billed by the hour, which this repo does
  not measure.
- The failure mode measured here is specific to agentgateway's `provider:
  ollama`, which speaks Ollama's OpenAI-compatible endpoint. Calling Ollama's
  native API directly, with `"think": false` honored, would very likely
  close most of this gap. That is a different, ungoverned path with no
  gateway budget, audit log or routing in front of it, which is the entire
  reason this repo put a gateway there in the first place.
- Raising `max_tokens` further would answer more cases for both Ollama models,
  at the cost of even worse p95 latency on the hardest ones. The ceiling
  used here was already chosen generously; this is a real tradeoff, not an
  oversight to tune away.

## License

Apache-2.0. `tasks.jsonl` is reused verbatim from
[themsquared/jev-benchmark](https://github.com/themsquared/jev-benchmark),
also Apache-2.0.
