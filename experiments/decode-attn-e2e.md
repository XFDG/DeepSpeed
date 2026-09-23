# Experiment: decode-attn-e2e

- **Status**: completed
- **Date**: 2026-09-21
- **Tmux**: session 0, window 5, pane 0

## Hardware

- Remote: root@connect.bjb1.seetacloud.com:15746 (autodl pro-7838522e6e9b)
- GPU: NVIDIA GeForce RTX 4080 SUPER, 1 (16 GB)
- CPU/RAM: 12 cores / 62 GB

## Environment

- Remote DeepSpeed: /root/DeepSpeed (worktree segment-ki deployed)
- vLLM: /root/vllm-env/bin/python (FLASH_ATTN backend)
- Model: Qwen/Qwen3.5-4B-Base bf16 (32 layers: 24 GDN + 8 full-attention, 16Q/4KV heads, head_dim 256)
- Note: FLA / causal-conv1d NOT installed on remote → GDN layers run the pure-torch fallback (dominant cost for both DS and vLLM)

## Code Changes (local .worktrees/segment-ki, deployed via chunked base64 over tmux)

- `deepspeed/module_inject/segment_ki.py` — added `_decode_attn_forward` (b=1 decode replacement for full-attention layers: verbatim HF q_proj 2×-wide gate split, q/k_norm, RoPE, KV update; SDPA replaced with `decode_attn` kernel reading write_pos; sigmoid-gate + o_proj) and `install_decode_attention(model, write_pos, cuda_op)` (structural detection + layer_types cross-check + kernel GQA/head_dim contract check; returns patched count)
- `deepspeed/runtime/rollout/hybrid_engine_rollout.py` — `_generate_graph` calls `install_decode_attention(module, write_pos, attn_op)` after DeepSpeedStaticCache setup, before warmup/graph capture
- `e2e_attn_test.py`, `baseline_e2e.py` (SDPA stub), `verify_attn_patch.py`, `deploy_attn.sh` — test scripts

### Transfer

- Method: chunked base64 via tmux send-keys (2000 B chunks), all 4 files md5-verified on remote

## Commands

1. E2E (kernel): `USE_KI=1 DS_QWEN2_INJECTION=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --standalone --nproc_per_node=1 /root/e2e_attn_test.py`
2. Verify: same env, `/root/verify_attn_patch.py` → PATCHED_LAYERS: 8, KERNEL_HAS_DECODE_ATTN: True
3. Baseline (SDPA): same env, `/root/baseline_e2e.py`
4. vLLM: `HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /root/vllm-env/bin/python /root/bench_vllm.py`

## Results (512 tokens, greedy, prompt "The capital of France is", 5 timed runs)

| Path | Olympic avg | tok/s | vs vLLM |
|---|---|---|---|
| DeepSpeed + decode_attn kernel | 7.558 s | **67.7** | 88.8% |
| DeepSpeed SDPA baseline | 7.701 s | 66.5 | 87.3% |
| vLLM (FLASH_ATTN) | 6.716 s | **76.2** | 100% |

- All times: DS-kernel ['7.543','7.586','7.556','7.559','7.558']; DS-SDPA ['7.700','7.729','7.713','7.690','7.685']; vLLM ['6.717','6.723','6.717','6.715','6.709']
- TEXT_OK: PASS in all DS runs (output starts " Paris. …")
- Kernel micro-bench (earlier session): 2.7–9.2× faster than SDPA mem_efficient per call

## Notes

- Only 8/32 layers are full-attention and attention is a minority of decode time (GDN torch fallback dominates), so the 2.7–9.2× per-call kernel win nets only +1.8% E2E (66.5 → 67.7 tok/s). Gap to vLLM (67.7 vs 76.2) is mostly the missing FLA/causal-conv1d fast path for GDN, not attention.
- Kernel graph-compat verified: same write_pos tensor object used by DeepSpeedStaticCache update and decode_step_graph; correct text over 512-token replays.

## Reproduction

1. `autodl-on` (retry on 无库存), SSH auto-connects in pane 0:5.0
2. Deploy segment_ki.py / hybrid_engine_rollout.py / fused_glu.cu / e2e_attn_test.py via chunked base64; `rm -rf ~/.cache/torch_extensions/*/fused_glu*`
3. Run commands 1–4 above; `shutdown now` when done

## Tier 1/2 follow-up (same instance, later session)

### Code Changes
- `deepspeed/module_inject/segment_ki.py` — `_decode_attn_forward`: optional Tier 2 fast path (`op.triple_gemv` replaces the three q/k/v `F.linear` calls; gate split unchanged; enabled when `self._ki_qkv_op` set, `DS_TIER2=1` default). New `_tier12_layer_forward` + `install_tier12_norm(model, cuda_op)`: at b=1/seq_len=1 the post-attention residual-add + RMSNorm pair runs through `op.fused_add_norm` (kernel mutates attn-out in-place into the residual stream and returns the normalized MLP input); pre-attention norm and post-MLP add stay native (no add to fuse there). RMSNorm scale precomputed as bf16 `1+w` persistent buffer (HF multiplies by 1+weight, kernel by its weight arg directly). `DS_TIER1` env kill-switch. All 32 decoder layers patched (structural detection: block_type + two layernorms).
- `deepspeed/runtime/rollout/hybrid_engine_rollout.py` — `_generate_graph` calls `install_tier12_norm(module, attn_op)` after `install_decode_attention`; stores `module._ki_tier12_patched`.
- `e2e_tier12_test.py`, `tier12_check.py` (48-token fast text check) — new test scripts.

### Debugging note
First run degenerated to " Paris Paris Paris..." — variable-binding bug: `fused_add_norm` returns the NORMALIZED tensor while mutating arg 1 in-place into the residual stream; the final add was going through the normalized tensor instead of the residual stream. Fixed by keeping both handles (`new_residual` = mutated arg, `mlp_input` = return value). Bisected via `DS_TIER1`/`DS_TIER2` env switches (Tier2 alone was always correct).

### Results (512 tok, greedy, 5 runs, Olympic avg)
| Path | Olympic avg | tok/s | vs vLLM |
|---|---|---|---|
| DeepSpeed + decode_attn + Tier1 + Tier2 | 7.402 s | **69.2** | 92.4% |
| (prior) DS + decode_attn only | 7.558 s | 67.7 | 88.8% |
| vLLM (FLASH_ATTN, same instance) | — | 74.9 (128-tok bench) | 100% |

- Tier1/2 gain: +2.2% (67.7 → 69.2); all-times ['7.387','7.387','7.387','7.434','7.433']
- TEXT_OK: PASS (' Paris.\n\nThe capital of the United States…'); TIER12_PATCHED: 32, ATTN/QKV_PATCHED: 8/8
- flake8 could not run in the local sandbox (multiprocessing SemLock denied); syntax-verified via ast.parse only

### Transfer
- chunked base64 via tmux: segment_ki.py (85b3aa8a8daf162891e02fe42ca894ad), hybrid_engine_rollout.py (646f02f506d66c7a8bcd7657a92dd75d), e2e_tier12_test.py (e7f25169d5ba7efdb6909b5099c0a0d8), tier12_check.py (7c101c9de72507929aad29a0a2304568) — all md5-verified
