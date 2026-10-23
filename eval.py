import os
import csv
import re
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from peft import PeftModel

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
from pycocoevalcap.cider.cider import Cider

from PRISM import DimensionAwarePRISM, DEFAULT_DIM_NAMES

hf_logging.set_verbosity_error()


DEFAULT_MODEL     = ""
DEFAULT_FEATURES  = ""
DEFAULT_REPORTCSV = ""
PRISM_CKPT         = ""
DEFAULT_LORA_DIR  = ""
DEFAULT_OUTCSV    = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = ""


BANNED_PREFIXES = []


def read_refs(csv_path: str):
    ref_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.reader(f, dialect)
        rows = list(reader)
        has_header = False
        if len(rows) > 0:
            first = rows[0]
            if len(first) >= 2 and (first[0].lower().startswith("video") or first[0].lower() == "id"):
                has_header = True
        for i, r in enumerate(rows):
            if i == 0 and has_header:
                continue
            if len(r) < 2:
                continue
            ref_map[r[0].strip()] = r[1].strip()
    return ref_map


def clean_output(text: str) -> str:
    t = text.strip()
    if PROMPT_TEMPLATE in t:
        t = t.split(PROMPT_TEMPLATE)[-1].strip()
    for pat in BANNED_PREFIXES:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    sents = re.split(r'(?<=[.!?])\s+', t)
    if sents:
        s0 = re.sub(r'^[\s;:,\.\-–—]+', '', sents[0])
        s0 = re.sub(r';{2,}', ';', s0)
        sents[0] = s0
    out = " ".join(sents).strip()
    out = out.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return out


def infer_feat_dim(feature_dir: str) -> int:

    for p in Path(feature_dir).glob("*.npy"):
        arr = np.load(p)
        if arr.ndim == 2:
            return int(arr.shape[1])
        elif arr.ndim == 1:
            return int(arr.shape[0])
    raise RuntimeError(f"No .npy found under {feature_dir}")


def extract_prism_state(ckpt):

    if not isinstance(ckpt, dict):
        return ckpt
    if "prism" in ckpt:
        return ckpt["prism"]
    for value in ckpt.values():
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value
    return ckpt


def load_prism(args, llm_hidden: int, feat_dim: int, device: str):

    if not (args.prism_ckpt and os.path.exists(args.prism_ckpt)):
        raise ValueError(f"PRISM checkpoint not found: {args.prism_ckpt}")

    ckpt = torch.load(args.prism_ckpt, map_location="cpu", weights_only=True)
    stage1_args = ckpt.get("args", {})

    def _a(name, fallback):
        return stage1_args[name] if name in stage1_args else fallback

    n_dim_groups = _a("n_dim_groups", args.n_dim_groups)

    prism = DimensionAwarePRISM(
        vis_dim=feat_dim,
        hidden=llm_hidden,
        n_dim_groups=n_dim_groups,
        queries_per_group=_a("queries_per_group", args.queries_per_group),
        n_free_queries=_a("n_free_queries",     args.n_free_queries),
        n_blocks=_a("n_blocks",                 args.n_blocks),
        n_heads=_a("n_heads",                   args.n_heads),
        dropout=0.0,
        ffn_ratio=_a("ffn_ratio",               args.ffn_ratio),
        use_mamba=_a("use_mamba",               args.use_mamba),
        mamba_layers=_a("mamba_layers",         args.mamba_layers),
        mamba_d_state=_a("mamba_d_state",       args.mamba_d_state),
        mamba_d_conv=_a("mamba_d_conv",         args.mamba_d_conv),
        mamba_expand=_a("mamba_expand",         args.mamba_expand),
        bidirectional=_a("mamba_bidir",         args.mamba_bidir),
        use_block_diag_mask=_a("use_block_diag_mask", True),
        dim_names=DEFAULT_DIM_NAMES[:n_dim_groups],
    ).to(device)

    state = extract_prism_state(ckpt)
    missing, unexpected = prism.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] PRISM missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] PRISM unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

    for p in prism.parameters():
        p.requires_grad = False
    prism.eval()

    print(f"[PRISM] loaded {args.prism_ckpt}")
    print(f"       (epoch={ckpt.get('epoch','?')}, val_lm={ckpt.get('val_lm','?')})")
    print(f"       n_dim_groups={n_dim_groups}  prefix_tokens={prism.prefix_tokens}")
    return prism


@torch.no_grad()
def build_inputs_prism_eval(prism, model, tokenizer, feature_path: str,
                             prompt: str, device: str):

    vis = np.load(feature_path).astype(np.float32)
    if vis.ndim == 1:
        vis = vis.reshape(1, -1)
    vis_t = torch.from_numpy(vis).unsqueeze(0).to(device).float()
    vis_mask = torch.ones(1, vis_t.size(1), dtype=torch.bool, device=device)

    token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    tok_emb = model.get_input_embeddings()(token_ids)

    prefix, _ = prism(vis_t, vis_mask, tok_emb, return_aux=False)

    inputs_embeds = torch.cat([prefix.to(tok_emb.dtype), tok_emb], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    return inputs_embeds, attn


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--feature_dir", type=str, default=DEFAULT_FEATURES)
    parser.add_argument("--report_csv", type=str, default=DEFAULT_REPORTCSV)
    parser.add_argument("--prism_ckpt", type=str, default=PRISM_CKPT, help="Path to Stage-1 best_prism.pt")
    parser.add_argument("--lora_dir", type=str, default=DEFAULT_LORA_DIR, help="Path to Stage-2 LoRA folder")
    parser.add_argument("--out_csv", type=str, default=DEFAULT_OUTCSV)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N samples (0 = all)")


    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=3)


    parser.add_argument("--bertscore_model", type=str, default="roberta-base")
    parser.add_argument("--bertscore_baseline", action="store_true")
    parser.add_argument("--rouge_type", type=str, default="rougeL")
    parser.add_argument("--medbert_model", type=str,
        default="Shushant/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ContaminationQAmodel_PubmedBERT")


    parser.add_argument("--n_dim_groups", type=int, default=5)
    parser.add_argument("--queries_per_group", type=int, default=6)
    parser.add_argument("--n_free_queries", type=int, default=15)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ffn_ratio", type=int, default=4)
    parser.add_argument("--use_mamba", action="store_true", default=True)
    parser.add_argument("--no_mamba", dest="use_mamba", action="store_false")
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_bidir", action="store_true", default=True)

    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    base_model.config.pad_token_id = tokenizer.pad_token_id


    print(f"Loading LoRA adapter from: {args.lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.eval()


    feat_dim = infer_feat_dim(args.feature_dir)
    print(f"[Init] Inferred visual feature dimension D={feat_dim}")
    prism = load_prism(args, base_model.config.hidden_size, feat_dim, DEVICE)
    prism_prefix_tokens = prism.prefix_tokens


    amp_dtype = None
    if args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        amp_dtype = torch.float16


    ref_map = read_refs(args.report_csv)
    video_ids = sorted(ref_map.keys())
    valid_ids = [vid for vid in video_ids
                 if (Path(args.feature_dir) / f"{vid}.npy").exists()]
    if args.limit and args.limit > 0:
        valid_ids = valid_ids[:args.limit]
    print(f"Evaluating {len(valid_ids)} samples ...")


    bleu_scorers = {
        f"BLEU{n}": BLEU(effective_order=True, max_ngram_order=n)
        for n in range(1, 5)
    }
    rouge = rouge_scorer.RougeScorer([args.rouge_type], use_stemmer=True)


    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=40,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=4,
        repetition_penalty=1.1,
    )
    if args.do_sample:
        gen_kwargs.update(dict(
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
        ))
    else:
        gen_kwargs.update(dict(
            do_sample=False,
            num_beams=args.num_beams,
            length_penalty=1.05,
            early_stopping=True,
        ))

    records = []
    for vid in valid_ids:
        ref = ref_map[vid]
        feature_path = str(Path(args.feature_dir) / f"{vid}.npy")

        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
            inputs_embeds, attn_mask = build_inputs_prism_eval(
                prism, model, tokenizer, feature_path, PROMPT_TEMPLATE, DEVICE,
            )
            output = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                **gen_kwargs,
            )

        pred_full = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = clean_output(pred_full)

        bleu_vals = [
            bleu_scorers[f"BLEU{n}"].sentence_score(pred, [ref]).score / 100.0
            for n in range(1, 5)
        ]
        rouge_val = rouge.score(ref, pred)[args.rouge_type].fmeasure
        meteor = single_meteor_score(word_tokenize(ref), word_tokenize(pred))

        records.append([vid, ref, pred, *bleu_vals, rouge_val, meteor])
        print(f"[{vid}] B1={bleu_vals[0]:.4f} B2={bleu_vals[1]:.4f} "
              f"B3={bleu_vals[2]:.4f} B4={bleu_vals[3]:.4f} "
              f"{args.rouge_type}={rouge_val:.4f} MT={meteor:.4f}")


    print("\n>>> Computing heavy metrics (CIDEr, BERTScore, Med-BERTScore) in batches...")
    refs_list = [r[1] for r in records]
    preds_list = [r[2] for r in records]

    gts = {i: [ref] for i, ref in enumerate(refs_list)}
    res = {i: [pred] for i, pred in enumerate(preds_list)}
    cider_scorer = Cider()
    _, cider_scores = cider_scorer.compute_score(gts, res)
    print("    [Done] CIDEr")

    print(f"    [Running] Standard BERTScore ({args.bertscore_model}) ...")
    _, _, std_f1_tensor = bert_score(
        preds_list, refs_list,
        lang="en",
        model_type=args.bertscore_model,
        rescale_with_baseline=args.bertscore_baseline,
        verbose=True,
        device=DEVICE,
        batch_size=32,
    )
    std_f1_scores = std_f1_tensor.cpu().numpy()

    print(f"    [Running] Med-BERTScore ({args.medbert_model}) ...")
    _, _, med_f1_tensor = bert_score(
        preds_list, refs_list,
        lang="en",
        model_type=args.medbert_model,
        rescale_with_baseline=False,
        verbose=True,
        device=DEVICE,
        batch_size=32,
        num_layers=12,
    )
    med_f1_scores = med_f1_tensor.cpu().numpy()


    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    base_cols = ["video_id", "reference", "prediction",
                 "BLEU1", "BLEU2", "BLEU3", "BLEU4",
                 args.rouge_type.upper(), "METEOR"]
    df = pd.DataFrame(records, columns=base_cols)
    df["BERTScore_F1"]    = std_f1_scores
    df["MedBERTScore_F1"] = med_f1_scores
    df["CIDEr"]           = cider_scores

    final_cols = base_cols + ["BERTScore_F1", "MedBERTScore_F1", "CIDEr"]
    df = df[final_cols]

    metric_cols = ["BLEU1", "BLEU2", "BLEU3", "BLEU4",
                   args.rouge_type.upper(), "METEOR",
                   "BERTScore_F1", "MedBERTScore_F1", "CIDEr"]
    n_samples = len(df)
    mean_row = {col: "" for col in final_cols}
    mean_row["video_id"] = "AVERAGE"
    mean_row["reference"] = f"N={n_samples}"
    mean_row["prediction"] = ""
    for col in metric_cols:
        mean_row[col] = float(df[col].mean())

    df_out = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 50)
    print(f"FINAL METRICS SUMMARY (N={n_samples})")
    print("=" * 50)
    for col in metric_cols:
        print(f"{col:18s}: {df[col].mean():.4f}")
    print("=" * 50)
    print(f"Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
