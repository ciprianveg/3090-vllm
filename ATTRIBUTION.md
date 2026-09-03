# Attribution

## Upstream projects

- **vLLM** — https://github.com/vllm-project/vllm (Apache-2.0)
- **wtdcode/vllm-backport PR #58** — DeepSeek-V4-Flash-Vision-Exp support with
  sm80 (Ampere) fixes; the base for the Docker image in this repo
  (branch `kaka86mm:pr/vision-sm80`, includes cherry-picks from
  vLLM PR #54566 for the vision model wrapper)
- **vLLM PR #52452** — "[Bugfix][Structured Output][Spec Decode] Validate accepted
  blocks before commit" — backported in `patches/structured_output_init.py` and the
  scheduler hunks of `patches/scheduler.py`

## Original fixes in this repo

- **Scheduler transition repair** (`patches/scheduler.py`, marked
  `Transition repair`): grants the missing bonus row when a chunked-prefilled
  request enters speculative decode, fixing the
  `assert num_scheduled_tokens >= num_logits` failure with grammar/tool-call
  requests (upstream issues vllm-project/vllm#49002, #49210).
- **10x RTX 3090 deployment recipes, memory budgeting, and benchmarks** for
  DeepSeek-V4-Flash-Vision-Exp on SM86.

## Model

- **DeepSeek-V4-Flash-Vision-Exp** — DeepSeek (weights distributed via Hugging Face;
  follow the model's own license terms).
