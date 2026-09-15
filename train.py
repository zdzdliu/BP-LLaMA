#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BP-LLaMA: Cuffless blood pressure estimation with a QLoRA fine-tuned Llama-3-8B.

The model is trained to generate MAP and PP values in text form
("Predicted_map: X.X mmHg / Predicted_pp: Y.Y mmHg"), which are then converted
back to SBP/DBP via MAP = DBP + PP/3 and PP = SBP - DBP.

Pipeline
--------
1. Load a 4-bit Llama-3-8B checkpoint via Unsloth and attach a QLoRA adapter.
2. Build an instruction prompt (three variants: ``basic`` / ``with_knowledge`` /
   ``with_knowledge_and_user_info``) from the preprocessed ``text`` field.
3. Fine-tune with TRL ``SFTTrainer`` using either:
   - ``--loss_mode completion``: completion-only loss (prompt tokens masked to -100), or
   - ``--loss_mode full``: full-sequence language-modeling loss.
4. Greedy-generate the MAP/PP completion, parse it, and report raw / calibrated
   (0.3*model + 0.7*calibration BP) / baseline-only metrics.

Data
----
The dataset directory should contain three JSON files:

    dataset/train_dataset.json
    dataset/valid_dataset.json
    dataset/test_dataset.json

Each JSON is a list of records with the fields ``text``, ``refsbp``, ``refdbp``,
``basesbp``, and ``basedbp`` (see the provided examples).

Usage
-----
  python train.py
  python train.py --train dataset/train_dataset.json --test dataset/test_dataset.json
"""

import argparse
import json
import os
import random
import re
import sys
import time

# Unsloth must be imported before transformers/trl.
from unsloth import FastLanguageModel  # noqa: E402  isort:skip

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

# Root of this repository (dataset/ and outputs/ live here).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# 0. Command-line interface
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="dataset/train_dataset.json",
                   help="Path to the training JSON.")
    p.add_argument("--valid", default="dataset/valid_dataset.json",
                   help="Path to the validation JSON (used for early stopping).")
    p.add_argument("--test", default="dataset/test_dataset.json",
                   help="Path to the test JSON.")
    p.add_argument("--prompt_type", default="with_knowledge_and_user_info",
                   choices=["basic", "with_knowledge", "with_knowledge_and_user_info"])

    p.add_argument("--model_dir", default=None,
                   help="Path to a local Llama-3-8B checkpoint (bnb 4-bit). "
                        "Defaults to the official HF repo if not set.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--tag", default=None, help="Extra suffix for the output directory.")

    # Loss mode: completion = answer-only loss; full = whole-sequence loss.
    p.add_argument("--loss_mode", default="completion", choices=["completion", "full"])

    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)

    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)

    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--infer_batch_size", type=int, default=32)
    p.add_argument("--limit_test", type=int, default=0,
                   help="If > 0, only run the first N test samples (smoke test).")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training and load the adapter already saved in output_dir.")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 1. Prompt construction (identical to the original notebook)
# =========================================================
def _strip_response(full_text):
    if "### Response:" in full_text:
        return full_text.split("### Response:")[0].strip()
    return full_text.strip()


def _extract_between(text, start, end_patterns):
    if start not in text:
        return ""
    sub = text.split(start, 1)[1]
    end_pos = len(sub)
    for pat in end_patterns:
        m = re.search(pat, sub)
        if m:
            end_pos = min(end_pos, m.start())
    return sub[:end_pos].strip()


def _extract_feature_list(text, feature_name):
    m = re.search(rf"{re.escape(feature_name)}:\s*(\[[^\]]*\])", text, flags=re.S)
    return m.group(1).strip() if m else "[]"


def _merge_feature_lists(*feature_lists):
    values = []
    for feat in feature_lists:
        values.extend(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", feat))
    return "[" + ", ".join(values) + "]"


def build_prompt(example, prompt_type):
    """Return the prompt without the answer (ends with '### Response:\\n')."""
    full_text = example.get("text", "")
    body = _strip_response(full_text)

    if prompt_type == "with_knowledge_and_user_info":
        return (body + "\n\n### Response:\n").strip() + "\n"

    instruction = _extract_between(body, "### Instruction:", [r"### Input:"])
    bp_knowledge = _extract_between(
        body, "### Input:", [r"Given the user's profile:", r"Cardiac output features:"])
    cardiac = _extract_feature_list(body, "Cardiac output features")
    svr = _extract_feature_list(body, "Systemic vascular resistance features")
    stiff = _extract_feature_list(body, "Arterial stiffness features")

    header = ("Below is an instruction that describes a task, paired with an input that "
              "provides further context. Write a response that appropriately completes "
              "the request.")

    if prompt_type == "basic":
        merged = _merge_feature_lists(cardiac, svr, stiff)
        prompt = (f"{header}\n\n### Instruction:\n{instruction}\n\n### Input:\n"
                  f"The physiological features: {merged}\n\n"
                  f"Based on this data, what would be the predicted MAP and PP values?\n\n"
                  f"### Response:\n")
    else:  # with_knowledge
        prompt = (f"{header}\n\n### Instruction:\n{instruction}\n\n### Input:\n{bp_knowledge}\n\n"
                  f"Cardiac output features: {cardiac}\n"
                  f"Systemic vascular resistance features: {svr}\n"
                  f"Arterial stiffness features: {stiff}\n\n"
                  f"Based on this data, what would be the predicted MAP and PP values?\n\n"
                  f"### Response:\n")
    return prompt


def extract_completion(example):
    """Extract the answer text after '### Response:'."""
    full_text = example.get("text", "")
    if "### Response:" in full_text:
        return full_text.split("### Response:")[1].strip()
    return ""


# =========================================================
# 2. Numeric parsing and conversion
# =========================================================
FLOAT_RE = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)"


def extract_map_pp(text):
    m1 = re.search(rf"Predicted_map\s*:\s*({FLOAT_RE})", text, flags=re.IGNORECASE)
    m2 = re.search(rf"Predicted_pp\s*:\s*({FLOAT_RE})", text, flags=re.IGNORECASE)
    return (float(m1.group(1)) if m1 else None, float(m2.group(1)) if m2 else None)


def map_pp_to_sbp_dbp(pred_map, pred_pp):
    """MAP = DBP + PP/3, PP = SBP - DBP  =>  SBP = MAP + 2/3*PP, DBP = MAP - 1/3*PP."""
    return pred_map + 2.0 * pred_pp / 3.0, pred_map - pred_pp / 3.0


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# =========================================================
# 3. Metrics
# =========================================================
def error_stats(ref, pred):
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    err = pred - ref
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "ME": float(np.mean(err)),
        "SD": float(np.std(err, ddof=1)) if len(err) > 1 else 0.0,
        "R": float(np.corrcoef(ref, pred)[0, 1]) if len(ref) > 1 else 0.0,
    }


def bhs_grade(ref, pred):
    """BHS grade based on the percentage of errors within 5/10/15 mmHg."""
    err = np.abs(np.asarray(pred, float) - np.asarray(ref, float))
    p5, p10, p15 = [float(np.mean(err <= t) * 100) for t in (5, 10, 15)]
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        grade = "A"
    elif p5 >= 50 and p10 >= 75 and p15 >= 90:
        grade = "B"
    elif p5 >= 40 and p10 >= 65 and p15 >= 85:
        grade = "C"
    else:
        grade = "D"
    return {"within5_pct": p5, "within10_pct": p10, "within15_pct": p15, "grade": grade}


def aami_check(stats):
    """AAMI/ANSI criterion: |ME| <= 5 and SD <= 8 is a pass."""
    return {"ME": stats["ME"], "SD": stats["SD"],
            "pass": bool(abs(stats["ME"]) <= 5.0 and stats["SD"] <= 8.0)}


def full_report(ref_sbp, ref_dbp, pred_sbp, pred_dbp, name):
    rep = {}
    for label, ref, pred in (("SBP", ref_sbp, pred_sbp), ("DBP", ref_dbp, pred_dbp)):
        st = error_stats(ref, pred)
        st["BHS"] = bhs_grade(ref, pred)
        st["AAMI"] = aami_check(st)
        rep[label] = st
    return {name: rep}


# =========================================================
# 4. Batched generation
# =========================================================
@torch.no_grad()
def _gen_uniform(model, tokenizer, id_lists, max_new_tokens):
    """Generate for a set of tokenized prompts that all share the same length (no padding)."""
    input_ids = torch.tensor(id_lists, device=model.device)
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = gen[:, input_ids.shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(new_ids, skip_special_tokens=True)]


@torch.no_grad()
def batch_generate(model, tokenizer, prompts, max_new_tokens, batch_size, max_seq_length):
    """
    Group prompts into equal-length buckets and batch each bucket, so no padding is
    ever introduced.

    Rationale: Unsloth 2025.10.8's batched generation corrupts left-padded samples
    (within a batch, the sample whose length differs from the others degrades to
    "_map: ... predicted_pp: ...", dropping the first token and changing case), which
    drives the parse-failure rate above 70%. Zero-padding via length bucketing keeps
    both batched speed and per-sample accuracy.
    """
    tokenizer.padding_side = "left"
    enc_all = [tokenizer(p, truncation=True, max_length=max_seq_length)["input_ids"]
               for p in prompts]

    buckets = {}
    for i, ids in enumerate(enc_all):
        buckets.setdefault(len(ids), []).append(i)
    print(f"  [infer] {len(prompts)} prompts -> {len(buckets)} equal-length buckets "
          f"(length {min(buckets)}~{max(buckets)})", flush=True)

    outs = [None] * len(prompts)
    done, t0 = 0, time.time()
    for _, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), batch_size):
            grp = idxs[s:s + batch_size]
            for i, txt in zip(grp, _gen_uniform(model, tokenizer,
                                                [enc_all[i] for i in grp], max_new_tokens)):
                outs[i] = txt
            done += len(grp)
            if done % (batch_size * 10) < batch_size or done == len(prompts):
                el = time.time() - t0
                print(f"  [infer] {done}/{len(prompts)}  {el:.0f}s  "
                      f"eta {el / done * (len(prompts) - done):.0f}s", flush=True)

    # Fallback: re-run any sample that still fails to parse, one at a time.
    retry = [i for i, t in enumerate(outs) if None in extract_map_pp(t)]
    if retry:
        print(f"  [infer] {len(retry)} samples failed to parse, retrying with batch=1", flush=True)
        for i in retry:
            outs[i] = _gen_uniform(model, tokenizer, [enc_all[i]], max_new_tokens)[0]
    return outs


# =========================================================
# 5. Main pipeline
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    if args.model_dir is None:
        # The official Llama-3-8B (bnb 4-bit) checkpoint must be available locally
        # or on the HF Hub. Set --model_dir to point at a local copy.
        args.model_dir = "unsloth/llama-3-8b-bnb-4bit"

    train_path = os.path.join(PROJECT_ROOT, args.train)
    valid_path = os.path.join(PROJECT_ROOT, args.valid)
    test_path = os.path.join(PROJECT_ROOT, args.test)
    for p in (train_path, valid_path, test_path):
        if not os.path.exists(p):
            sys.exit(f"Data file not found: {p}")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        stem = os.path.splitext(os.path.basename(args.test))[0]
        sub = f"{stem}_{args.prompt_type}" + (f"_{args.tag}" if args.tag else "")
        out_dir = os.path.join(PROJECT_ROOT, "outputs", sub)
    os.makedirs(out_dir, exist_ok=True)

    run_cfg = vars(args) | {"output_dir": out_dir, "train_path": train_path,
                            "valid_path": valid_path, "test_path": test_path}
    print("=" * 80)
    print(json.dumps(run_cfg, indent=2, ensure_ascii=False))
    print("=" * 80, flush=True)

    # ---------- Model ----------
    # With --skip_train, load the adapter already saved in out_dir and only re-infer.
    resume_adapter = args.skip_train and os.path.exists(
        os.path.join(out_dir, "adapter_config.json"))
    load_from = out_dir if resume_adapter else args.model_dir
    if resume_adapter:
        print(f"[skip_train] Loading trained adapter from: {load_from}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_from, max_seq_length=args.max_seq_length, dtype=None,
        load_in_4bit=True, load_in_8bit=False, full_finetuning=False, local_files_only=False)
    if not resume_adapter:
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=args.seed,
            use_rslora=False, loftq_config=None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Data ----------
    raw_train = load_dataset("json", data_files=train_path, split="train")
    raw_valid = load_dataset("json", data_files=valid_path, split="train")
    raw_test = load_dataset("json", data_files=test_path, split="train")

    # Unsloth's patched SFTTrainer does not accept TRL's prompt/completion columns
    # (it requires a formatting_func), so completion-only loss is implemented by
    # tokenizing manually and masking the prompt tokens to -100; Unsloth then
    # switches to DataCollatorForSeq2Seq automatically when a labels column exists.
    def to_tokenized(ex):
        prompt = build_prompt(ex, args.prompt_type)
        completion = extract_completion(ex)
        p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        c_ids = tokenizer(completion + tokenizer.eos_token,
                          add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[:args.max_seq_length]
        labels = ([-100] * len(p_ids) + c_ids)[:args.max_seq_length]
        return {"input_ids": ids, "labels": labels}

    def to_text(ex):
        text = build_prompt(ex, args.prompt_type) + extract_completion(ex)
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        return {"text": text}

    mapper = to_tokenized if args.loss_mode == "completion" else to_text

    fit_txt = raw_train.map(mapper, remove_columns=raw_train.column_names)
    val_txt = raw_valid.map(mapper, remove_columns=raw_valid.column_names)
    print(f"Train: {len(fit_txt)} | Val: {len(val_txt)} | Test: {len(raw_test)}")
    if args.loss_mode == "completion":
        ex0 = fit_txt[0]
        n_sup = sum(1 for x in ex0["labels"] if x != -100)
        print(f"[sample] len={len(ex0['input_ids'])} supervised_tokens={n_sup}")
        print("[supervised text]",
              repr(tokenizer.decode([x for x in ex0["labels"] if x != -100])))
    else:
        print(f"[sample text]\n{fit_txt[0]['text'][:600]}\n...")
    sys.stdout.flush()

    # ---------- Training ----------
    if not args.skip_train:
        sft_kwargs = dict(
            output_dir=out_dir,
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",
            seed=args.seed,
            report_to="none",
        )
        if args.loss_mode == "full":
            sft_kwargs["dataset_text_field"] = "text"

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,     # TRL 0.23: the old tokenizer= arg was removed.
            train_dataset=fit_txt,
            eval_dataset=val_txt,
            args=SFTConfig(**sft_kwargs),
        )
        train_result = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        print("Training finished:", train_result, flush=True)
        with open(os.path.join(out_dir, "train_result.json"), "w") as f:
            json.dump({"metrics": train_result.metrics,
                       "log_history": trainer.state.log_history}, f, indent=2)

    # ---------- Inference ----------
    FastLanguageModel.for_inference(model)
    model.eval()

    test_items = list(raw_test)
    if args.limit_test > 0:
        test_items = test_items[:args.limit_test]
    prompts = [build_prompt(it, args.prompt_type) for it in test_items]
    gen_texts = batch_generate(model, tokenizer, prompts, args.max_new_tokens,
                               args.infer_batch_size, args.max_seq_length)

    results, rows = [], []
    n_invalid = 0
    for i, (it, prompt, gen) in enumerate(zip(test_items, prompts, gen_texts)):
        pm, pp = extract_map_pp(gen)
        rec = {"index": i, "generated_output": gen, "pred_map": pm, "pred_pp": pp,
               "ref_sbp": safe_float(it.get("refsbp")), "ref_dbp": safe_float(it.get("refdbp")),
               "base_sbp": safe_float(it.get("basesbp")), "base_dbp": safe_float(it.get("basedbp"))}
        for extra in ("subid", "phase"):
            if extra in it:
                rec[extra] = it.get(extra)

        if pm is None or pp is None:
            n_invalid += 1
            rec.update(pred_sbp=None, pred_dbp=None, valid=False)
        else:
            sbp, dbp = map_pp_to_sbp_dbp(pm, pp)
            rec.update(pred_sbp=round(sbp, 3), pred_dbp=round(dbp, 3), valid=True)
            if None not in (rec["ref_sbp"], rec["ref_dbp"]):
                rows.append(rec)
        results.append(rec)

    # ---------- Metrics ----------
    csv_name = os.path.splitext(os.path.basename(args.test))[0] + "_" + args.prompt_type + ".csv"
    df = pd.DataFrame({
        "estsbp": [r["pred_sbp"] for r in rows], "estdbp": [r["pred_dbp"] for r in rows],
        "refsbp": [r["ref_sbp"] for r in rows], "refdbp": [r["ref_dbp"] for r in rows],
        "basesbp": [r["base_sbp"] for r in rows], "basedbp": [r["base_dbp"] for r in rows],
    })
    df.to_csv(os.path.join(out_dir, csv_name), index=False)

    metrics = {"config": run_cfg,
               "n_test_total": len(test_items), "n_valid": len(results) - n_invalid,
               "n_invalid": n_invalid,
               "valid_ratio": (len(results) - n_invalid) / max(len(results), 1)}

    if rows:
        ref_sbp = np.array([r["ref_sbp"] for r in rows], dtype=float)
        ref_dbp = np.array([r["ref_dbp"] for r in rows], dtype=float)
        est_sbp = np.array([r["pred_sbp"] for r in rows], dtype=float)
        est_dbp = np.array([r["pred_dbp"] for r in rows], dtype=float)
        base_sbp = np.array([r["base_sbp"] for r in rows], dtype=float)
        base_dbp = np.array([r["base_dbp"] for r in rows], dtype=float)
        ref_pp = ref_sbp - ref_dbp
        ref_map = ref_dbp + ref_pp / 3.0

        metrics.update(full_report(ref_sbp, ref_dbp, est_sbp, est_dbp, "raw"))
        metrics.update(full_report(ref_sbp, ref_dbp,
                                   est_sbp * 0.3 + base_sbp * 0.7,
                                   est_dbp * 0.3 + base_dbp * 0.7, "calibrated_0.3_0.7"))
        metrics.update(full_report(ref_sbp, ref_dbp, base_sbp, base_dbp, "baseline_only"))
        metrics["MAP"] = error_stats(ref_map, [r["pred_map"] for r in rows])
        metrics["PP"] = error_stats(ref_pp, [r["pred_pp"] for r in rows])

    with open(os.path.join(out_dir, "test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("=" * 80)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
