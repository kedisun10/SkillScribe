import os
import argparse
import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()

from PRISM import DimensionAwarePRISM, DEFAULT_DIM_NAMES
from train_adapter import (
    FeatureReportDataset,
    BatchCollator,
    read_reports,
    build_inputs_prism,
    compute_loss_with_label_smoothing,
    save_config_json,
    save_split_json,
    load_grouped_split,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_params(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot


DEFAULT_MODEL = ""
FEATURE_DIR   = ""
REPORT_CSV    = ""
PRISM_CKPT     = ""
OUT_DIR       = ""
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

PROMPT_TEMPLATE = ""


def build_datasets(tokenizer, args, report_map):
    train_ds = FeatureReportDataset(
        feature_dir=args.feature_dir, report_map=report_map, tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE, max_tokens=args.max_tokens,
        augment=True, aug_noise_std=args.aug_noise_std,
    )
    val_ds = FeatureReportDataset(
        feature_dir=args.feature_dir, report_map=report_map, tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE, max_tokens=args.max_tokens,
        augment=False, aug_noise_std=0.0,
    )
    return train_ds, val_ds


def resolve_split(full_ds, args):
    split_json = Path(args.prism_ckpt).parent / "split.json"
    train_idx, val_idx, _ = load_grouped_split(full_ds, str(split_json))
    print(f"[Split] Loading the Stage 1 fold from {split_json}")
    print(f"[Split] train={len(train_idx)}  val={len(val_idx)}  total={len(full_ds)}")
    return train_idx, val_idx


def extract_prism_state(ckpt):

    if not isinstance(ckpt, dict):
        return ckpt
    if "prism" in ckpt:
        return ckpt["prism"]
    for value in ckpt.values():
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value
    return ckpt


def load_prism(args, llm_hidden, feat_dim):

    if not (args.prism_ckpt and os.path.exists(args.prism_ckpt)):
        raise ValueError(
            f"Stage 1 PRISM checkpoint not found: {args.prism_ckpt}"
        )
    ckpt = torch.load(args.prism_ckpt, map_location="cpu", weights_only=True)


    stage1_args = ckpt.get("args", {})
    def _arg(name, fallback):
        return stage1_args[name] if name in stage1_args else fallback

    prism = DimensionAwarePRISM(
        vis_dim=feat_dim,
        hidden=llm_hidden,
        n_dim_groups=_arg("n_dim_groups",        args.n_dim_groups),
        queries_per_group=_arg("queries_per_group", args.queries_per_group),
        n_free_queries=_arg("n_free_queries",    args.n_free_queries),
        n_blocks=_arg("n_blocks",                args.n_blocks),
        n_heads=_arg("n_heads",                  args.n_heads),
        dropout=_arg("dropout",                  args.dropout),
        ffn_ratio=_arg("ffn_ratio",              args.ffn_ratio),
        use_mamba=_arg("use_mamba",              args.use_mamba),
        mamba_layers=_arg("mamba_layers",        args.mamba_layers),
        mamba_d_state=_arg("mamba_d_state",      args.mamba_d_state),
        mamba_d_conv=_arg("mamba_d_conv",        args.mamba_d_conv),
        mamba_expand=_arg("mamba_expand",        args.mamba_expand),
        bidirectional=_arg("mamba_bidir",        args.mamba_bidir),
        use_block_diag_mask=_arg("use_block_diag_mask", True),
        dim_names=DEFAULT_DIM_NAMES[:_arg("n_dim_groups", args.n_dim_groups)],
    ).to(args.device)

    state = extract_prism_state(ckpt)
    missing, unexpected = prism.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] PRISM missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] PRISM unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    print(f"[PRISM] Loaded Stage 1 weights (epoch={ckpt.get('epoch','?')}, "
          f"val_lm={ckpt.get('val_lm','?')}). prefix_tokens={prism.prefix_tokens}")


    for p in prism.parameters():
        p.requires_grad = False
    prism.eval()
    return prism


def run_training(args):
    save_config_json(args.out_dir, args)

    print(f"Loading model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)


    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lcfg)


    amp_dtype = None
    if args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        amp_dtype = torch.float16


    report_map = read_reports(args.report_csv)
    train_ds, val_ds = build_datasets(tokenizer, args, report_map)
    print(f"Matched {len(train_ds)} feature-report pairs")
    assert train_ds.video_ids == val_ds.video_ids, \
        "train_ds and val_ds video_ids must align — check FeatureReportDataset sort order."

    train_idx, val_idx = resolve_split(train_ds, args)
    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(val_ds,   val_idx)

    collator = BatchCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=collator)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collator)


    sample_path = Path(args.feature_dir) / f"{train_ds.video_ids[0]}.npy"
    feat_dim = int(np.load(str(sample_path)).shape[1])
    prism = load_prism(args, model.config.hidden_size, feat_dim)

    tr, _ = count_trainable_params(model)
    print(f"LoRA trainable params (LLM): {tr/1e6:.2f}M  | PRISM frozen.")


    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )


    best_dir = os.path.join(args.out_dir, "lora_best")
    last_dir = os.path.join(args.out_dir, "lora_last")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(last_dir, exist_ok=True)

    with open(Path(args.prism_ckpt).parent / "split.json", "r", encoding="utf-8") as f:
        split_info = json.load(f)
    save_split_json(
        args.out_dir,
        train_ids=[train_ds.video_ids[i] for i in train_idx],
        val_ids=[train_ds.video_ids[i] for i in val_idx],
        metadata=split_info,
    )

    log_path = os.path.join(args.out_dir, "log.csv")
    log_fields = ["epoch", "lr", "train_lm", "train_aux", "val_lm", "val_aux", "val_acc"]
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(log_fields)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    best_val = float("inf")
    patience_left = args.early_stop_patience


    aux_on = (args.lam_cls > 0) or (args.lam_orth > 0) or (args.lam_div > 0)
    if not aux_on:
        print("[Aux] All aux weights = 0 — skipping aux computation in forward.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        prism.eval()

        run_lm, run_aux = 0.0, 0.0
        n_steps = 0
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            ctx = (torch.cuda.amp.autocast(dtype=amp_dtype)
                   if amp_dtype is not None else nullcontext())
            with ctx:
                inputs_embeds, attn_mask, labels, _, _, aux, vmask = \
                    build_inputs_prism(prism, model, tokenizer, batch,
                                        args.device, return_aux=aux_on)
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    use_cache=False,
                    return_dict=True,
                )
                lm_loss = compute_loss_with_label_smoothing(
                    outputs.logits, labels, args.label_smoothing,
                )

                aux_total = lm_loss.new_zeros(())
                if aux_on:
                    if args.lam_cls > 0:
                        aux_total = aux_total + args.lam_cls * prism.aux_classification_loss(aux["logits"])
                    if args.lam_orth > 0:
                        aux_total = aux_total + args.lam_orth * prism.orth_penalty()
                    if args.lam_div > 0:
                        aux_total = aux_total + args.lam_div * prism.attn_diversity_loss(aux["attn_maps"], vmask)

                loss = lm_loss + aux_total

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            run_lm  += lm_loss.item()
            run_aux += aux_total.item() if aux_on else 0.0
            n_steps += 1
            pbar.set_postfix(lm=f"{lm_loss.item():.3f}",
                             aux=f"{aux_total.item():.3f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train_lm  = run_lm  / max(1, n_steps)
        avg_train_aux = run_aux / max(1, n_steps)


        model.eval()
        v_lm, v_aux, v_acc = 0.0, 0.0, 0.0
        v_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate", leave=False):
                ctx = (torch.cuda.amp.autocast(dtype=amp_dtype)
                       if amp_dtype is not None else nullcontext())
                with ctx:
                    inputs_embeds, attn_mask, labels, _, _, aux, vmask = \
                        build_inputs_prism(prism, model, tokenizer, batch,
                                            args.device, return_aux=True)
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    vloss = compute_loss_with_label_smoothing(
                        outputs.logits, labels, args.label_smoothing,
                    )
                    ce = prism.aux_classification_loss(aux["logits"])
                    acc = prism.aux_classification_accuracy(aux["logits"])["overall"]

                v_lm  += vloss.item()
                v_aux += ce.item()
                v_acc += acc
                v_steps += 1

        avg_val_lm  = v_lm  / max(1, v_steps)
        avg_val_aux = v_aux / max(1, v_steps)
        avg_val_acc = v_acc / max(1, v_steps)

        cur_lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch:03d}] train_lm={avg_train_lm:.4f} "
              f"train_aux={avg_train_aux:.4f}  |  "
              f"val_lm={avg_val_lm:.4f} val_ce={avg_val_aux:.4f} val_acc={avg_val_acc:.3f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{cur_lr:.8e}",
                f"{avg_train_lm:.6f}", f"{avg_train_aux:.6f}",
                f"{avg_val_lm:.6f}",   f"{avg_val_aux:.6f}",
                f"{avg_val_acc:.4f}",
            ])

        model.save_pretrained(last_dir)
        if avg_val_lm < best_val:
            best_val = avg_val_lm
            model.save_pretrained(best_dir)
            print(f"[BEST] Updated (val_lm={best_val:.4f})")
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] No improvement for {args.early_stop_patience} checks.")
                break

    return best_dir


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--feature_dir", type=str, default=FEATURE_DIR)
    parser.add_argument("--report_csv", type=str, default=REPORT_CSV)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--device", type=str, default=DEVICE)


    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=1024)


    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    parser.add_argument("--target_modules", type=str, default=DEFAULT_TARGET_MODULES)


    parser.add_argument("--aug_noise_std", type=float, default=0.02)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")


    parser.add_argument("--prism_ckpt", type=str, default=PRISM_CKPT,
                        help="Path to Stage 1 best_prism.pt")
    parser.add_argument("--n_dim_groups", type=int, default=5)
    parser.add_argument("--queries_per_group", type=int, default=6)
    parser.add_argument("--n_free_queries", type=int, default=15)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ffn_ratio", type=int, default=4)

    parser.add_argument("--use_mamba", action="store_true", default=True)
    parser.add_argument("--no_mamba", dest="use_mamba", action="store_false")
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_bidir", action="store_true", default=True)


    parser.add_argument("--lam_cls",  type=float, default=0.0)
    parser.add_argument("--lam_orth", type=float, default=0.0)
    parser.add_argument("--lam_div",  type=float, default=0.0)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    best_dir = run_training(args)
    print(f"\n[DONE] best LoRA adapter at: {best_dir}")


if __name__ == "__main__":
    main()
