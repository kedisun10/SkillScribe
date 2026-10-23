import os
import csv
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()

from PRISM import DimensionAwarePRISM, DEFAULT_DIM_NAMES


DEFAULT_MODEL = ""
FEATURE_DIR = ""
REPORT_CSV = ""
OUT_DIR = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = ""


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_reports(csv_path: str) -> Dict[str, str]:
    res = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.reader(f, dialect)
        rows = list(reader)
        first = rows[0]
        has_header = len(first) >= 2 and (
            first[0].lower() in ("video", "id") or first[1].lower().startswith("free")
        )
        for i, r in enumerate(rows):
            if i == 0 and has_header:
                continue
            if len(r) < 2:
                continue
            res[str(r[0]).strip()] = r[1].strip()
    return res


def save_config_json(out_dir: str, args: argparse.Namespace, filename: str = "config.json"):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            cfg[k] = v
        except (TypeError, ValueError):
            cfg[k] = str(v)
    cfg["_saved_at"] = datetime.now().isoformat(timespec="seconds")
    cfg_path = os.path.join(out_dir, filename)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[Config] Saved run config to {cfg_path}")
    return cfg_path


def save_split_json(out_dir: str,
                    train_ids: Sequence[str],
                    val_ids: Sequence[str],
                    filename: str = "split.json", metadata=None):
    os.makedirs(out_dir, exist_ok=True)
    split_path = os.path.join(out_dir, filename)
    info = dict(metadata or {})
    info.update({
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
    })
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[Split] Saved split ({len(train_ids)}/{len(val_ids)}) to {split_path}")
    return split_path


def load_grouped_split(dataset, split_path: str):
    if not split_path or not Path(split_path).is_file():
        raise FileNotFoundError(
            "Provide --split_json with the participant-grouped development-fold "
            "partition. Dataset split files will be released separately."
        )
    with open(split_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    partitions = {}
    for key in ("train_ids", "val_ids", "test_ids"):
        values = info.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list of video IDs")
        ids = [str(value).strip() for value in values]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate video IDs in {key}")
        partitions[key] = set(ids)
        info[key] = ids
    train_ids = partitions["train_ids"]
    val_ids = partitions["val_ids"]
    test_ids = partitions["test_ids"]
    if not train_ids or not val_ids:
        raise ValueError("The fold must contain non-empty training and validation IDs")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Training, validation and held-out test IDs must be disjoint")
    available = set(dataset.video_ids)
    missing = (train_ids | val_ids) - available
    if missing:
        raise ValueError(f"Missing feature-report pairs for split IDs: {sorted(missing)[:10]}")
    if available != train_ids | val_ids:
        raise ValueError("The development feature directory must match train_ids + val_ids")
    participants = info.get("participant_by_video")
    if participants is not None:
        if not isinstance(participants, dict):
            raise ValueError("participant_by_video must map video IDs to participant IDs")
        participants = {str(k): str(v).strip() for k, v in participants.items()}
        if any(not participants.get(vid) for vid in train_ids | val_ids | test_ids):
            raise ValueError("Missing participant IDs in participant_by_video")
        train_groups = {participants[vid] for vid in train_ids}
        val_groups = {participants[vid] for vid in val_ids}
        test_groups = {participants[vid] for vid in test_ids}
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise ValueError("Participant overlap between training, validation or test sets")
        if len(train_groups) != 8 or len(val_groups) != 2:
            raise ValueError("S-Skill development folds require 8 training and 2 validation participants")
        if test_ids and (len(test_groups) != 1 or len(test_ids) != 20):
            raise ValueError("S-Skill uses 20 test recordings from one held-out participant")
    train_idx = [i for i, vid in enumerate(dataset.video_ids) if vid in train_ids]
    val_idx = [i for i, vid in enumerate(dataset.video_ids) if vid in val_ids]
    return train_idx, val_idx, info


def routing_diversity_loss(route_global: torch.Tensor) -> torch.Tensor:
    B, G = route_global.shape
    if B < 2:
        return route_global.new_zeros(())
    p = route_global.float()
    eps = 1e-8
    H_per  = -(p * (p + eps).log()).sum(dim=-1).mean()
    mean_p = p.mean(dim=0)
    H_mean = -(mean_p * (mean_p + eps).log()).sum()
    return -(H_mean - H_per).to(route_global.dtype)


class FeatureReportDataset(Dataset):
    def __init__(self, feature_dir: str, report_map: Dict[str, str], tokenizer,
                 prompt_template: str, max_tokens: int,
                 augment: bool = False, aug_noise_std: float = 0.0):
        self.feature_dir = Path(feature_dir)
        self.report_map = report_map
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.augment = augment
        self.aug_noise_std = float(aug_noise_std)

        self.video_ids: List[str] = []
        for vid in report_map:
            if (self.feature_dir / f"{vid}.npy").exists():
                self.video_ids.append(vid)
        self.video_ids.sort()

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        feat_np = np.load(str(self.feature_dir / f"{vid}.npy")).astype(np.float32)

        feat_t = torch.from_numpy(feat_np)
        if self.augment and self.aug_noise_std > 0:
            feat_t = feat_t + torch.randn_like(feat_t) * self.aug_noise_std

        report_text = self.report_map[vid]
        full_text = self.prompt_template + report_text
        tokenized_full = self.tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=self.max_tokens,
        )
        input_ids = tokenized_full["input_ids"].squeeze(0)

        eos_id = self.tokenizer.eos_token_id
        if input_ids[-1].item() != eos_id:
            input_ids = torch.cat(
                [input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)],
                dim=0,
            )

        tokenized_prompt = self.tokenizer(
            self.prompt_template, return_tensors="pt",
            truncation=True, max_length=self.max_tokens,
        )
        prompt_len = tokenized_prompt["input_ids"].size(1)

        return {
            "video_id": vid,
            "feat_seq": feat_t,
            "input_ids": input_ids,
            "prompt_len": prompt_len,
        }


class BatchCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        B = len(batch)
        input_ids_list = [b["input_ids"] for b in batch]
        lengths = [x.size(0) for x in input_ids_list]
        max_len = max(lengths)
        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        for i, x in enumerate(input_ids_list):
            input_ids[i, :x.size(0)] = x
        prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)
        ids = [b["video_id"] for b in batch]


        seqs = [b["feat_seq"] for b in batch]
        S_max = max(x.size(0) for x in seqs)
        D = seqs[0].size(1)
        feat_seq = torch.zeros(B, S_max, D, dtype=seqs[0].dtype)
        feat_seq_mask = torch.zeros(B, S_max, dtype=torch.bool)
        for i, x in enumerate(seqs):
            S = x.size(0)
            feat_seq[i, :S, :] = x
            feat_seq_mask[i, :S] = True

        return {
            "video_id": ids,
            "input_ids": input_ids,
            "input_lens": torch.tensor(lengths, dtype=torch.long),
            "prompt_lens": prompt_lens,
            "feat_seq": feat_seq,
            "feat_seq_mask": feat_seq_mask,
        }


def compute_loss_with_label_smoothing(logits, labels, label_smoothing: float,
                                      ignore_index: int = -100):
    vocab_size = logits.size(-1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                   label_smoothing=label_smoothing)
    return loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))


def build_inputs_prism(prism: DimensionAwarePRISM, llm, tokenizer,
                        batch, device, return_aux: bool = True,
                        ignore_index: int = -100):

    vis_seq  = batch["feat_seq"].to(device).float()        
    vis_mask = batch["feat_seq_mask"].to(device).bool()    

    input_ids   = batch["input_ids"].to(device)
    lens        = batch["input_lens"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)
    B, Lmax = input_ids.size()


    with torch.no_grad():
        token_embeds = llm.get_input_embeddings()(input_ids)
        max_p = prompt_lens.max().item()
        txt_tokens = token_embeds.new_zeros(B, max_p, token_embeds.size(-1))
        for i in range(B):
            p = prompt_lens[i].item()
            txt_tokens[i, :p, :] = token_embeds[i, :p, :]


    prefix, aux = prism(vis_seq, vis_mask, txt_tokens, return_aux=return_aux)


    P = prism.prefix_tokens
    dtype = token_embeds.dtype
    inputs_embeds = torch.cat([prefix.to(dtype), token_embeds], dim=1)

    attn_mask = torch.zeros(B, P + Lmax, dtype=torch.long, device=device)
    labels = torch.full((B, P + Lmax), ignore_index, dtype=torch.long, device=device)
    for i in range(B):
        eff_len = P + lens[i].item()
        attn_mask[i, :eff_len] = 1
        P_i = prompt_lens[i].item()
        L_i = lens[i].item()
        if P_i < L_i:
            labels[i, P + P_i: P + L_i] = input_ids[i, P_i:L_i]

    return inputs_embeds, attn_mask, labels, prefix, token_embeds, aux, vis_mask


def compute_aux_loss(prism, aux, vis_mask, args):

    ce   = prism.aux_classification_loss(aux["logits"])
    orth = prism.orth_penalty()
    div  = prism.attn_diversity_loss(aux["attn_maps"], vis_mask)
    return ce, orth, div


def aux_warmup_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    if epoch >= warmup_epochs:
        return 1.0
    return float(epoch) / float(warmup_epochs)


def train_loop(prism: DimensionAwarePRISM, llm, tokenizer,
               train_loader, val_loader, args):

    llm.eval()
    os.makedirs(args.out_dir, exist_ok=True)


    mamba_params, other_params = [], []
    for name, p in prism.named_parameters():
        if not p.requires_grad:
            continue
        if "temporal_encoder" in name or "post_mamba_norm" in name:
            mamba_params.append(p)
        else:
            other_params.append(p)
    print(f"[Optim] mamba params: {sum(p.numel() for p in mamba_params)/1e6:.2f}M, "
          f"other params: {sum(p.numel() for p in other_params)/1e6:.2f}M")

    param_groups = [{"params": other_params, "lr": args.lr}]
    if mamba_params:
        param_groups.append({"params": mamba_params, "lr": args.lr * args.mamba_lr_mult})
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    trainable_params = other_params + mamba_params

    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )


    log_path = os.path.join(args.out_dir, "log.csv")
    log_fields = [
        "epoch", "lr", "aux_scale",
        "train_lm", "train_ce", "train_orth", "train_div", "train_acc",
        "val_lm",   "val_ce",   "val_acc",
    ]

    for d in prism.DIM_NAMES:
        log_fields.append(f"train_acc_{d}")
    for d in prism.DIM_NAMES:
        log_fields.append(f"val_acc_{d}")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(log_fields)

    best_val = float("inf")
    patience_left = args.early_stop_patience
    best_path = os.path.join(args.out_dir, "best_prism.pt")
    last_path = os.path.join(args.out_dir, "last_prism.pt")

    for epoch in range(1, args.epochs + 1):
        aux_scale = aux_warmup_scale(epoch - 1, args.aux_warmup_epochs)


        prism.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        run_lm, run_ce, run_orth, run_div = 0.0, 0.0, 0.0, 0.0
        run_acc_overall = 0.0
        run_acc_per_dim = {d: 0.0 for d in prism.DIM_NAMES}
        n_steps = 0

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inputs_embeds, attn_mask, labels, prefix_emb, token_emb, aux, vmask = \
                    build_inputs_prism(prism, llm, tokenizer, batch, args.device,
                                        return_aux=True)
                outputs = llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    use_cache=False,
                    return_dict=True,
                )
                lm_loss = compute_loss_with_label_smoothing(
                    outputs.logits, labels, args.label_smoothing,
                )

                ce_loss, orth_loss, div_loss = compute_aux_loss(prism, aux, vmask, args)

                aux_combined = (
                    args.lam_cls  * ce_loss   +
                    args.lam_orth * orth_loss +
                    args.lam_div  * div_loss
                )
                if args.lam_route > 0 and aux.get("route_global") is not None:
                    route_loss = routing_diversity_loss(aux["route_global"])
                    aux_combined = aux_combined + args.lam_route * route_loss

                total = lm_loss + aux_scale * aux_combined


                if args.prefix_norm_lambda > 0:
                    norm_reg = (prefix_emb ** 2).mean()
                    total = total + args.prefix_norm_lambda * norm_reg

            total = total.float()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()


            with torch.no_grad():
                acc_dict = prism.aux_classification_accuracy(aux["logits"])
            run_lm   += lm_loss.item()
            run_ce   += ce_loss.item()
            run_orth += orth_loss.item()
            run_div  += div_loss.item()
            run_acc_overall += acc_dict["overall"]
            for d in prism.DIM_NAMES:
                run_acc_per_dim[d] += acc_dict[f"acc_{d}"]
            n_steps += 1

            pbar.set_postfix(
                lm=f"{lm_loss.item():.3f}",
                ce=f"{ce_loss.item():.3f}",
                div=f"{div_loss.item():.3f}",
                acc=f"{acc_dict['overall']:.2f}",
            )

        avg_lm   = run_lm / max(1, n_steps)
        avg_ce   = run_ce / max(1, n_steps)
        avg_orth = run_orth / max(1, n_steps)
        avg_div  = run_div / max(1, n_steps)
        avg_acc  = run_acc_overall / max(1, n_steps)
        avg_acc_per_dim = {d: v / max(1, n_steps) for d, v in run_acc_per_dim.items()}


        prism.eval()
        v_lm, v_ce, v_acc = 0.0, 0.0, 0.0
        v_acc_per_dim = {d: 0.0 for d in prism.DIM_NAMES}
        v_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate", leave=False):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inputs_embeds, attn_mask, labels, _, _, aux, vmask = \
                        build_inputs_prism(prism, llm, tokenizer, batch, args.device,
                                            return_aux=True)
                    outputs = llm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    lm_loss = compute_loss_with_label_smoothing(
                        outputs.logits, labels, args.label_smoothing,
                    )
                    ce_loss = prism.aux_classification_loss(aux["logits"])
                    acc_dict = prism.aux_classification_accuracy(aux["logits"])

                v_lm  += lm_loss.item()
                v_ce  += ce_loss.item()
                v_acc += acc_dict["overall"]
                for d in prism.DIM_NAMES:
                    v_acc_per_dim[d] += acc_dict[f"acc_{d}"]
                v_steps += 1

        avg_v_lm  = v_lm  / max(1, v_steps)
        avg_v_ce  = v_ce  / max(1, v_steps)
        avg_v_acc = v_acc / max(1, v_steps)
        avg_v_acc_per_dim = {d: v / max(1, v_steps) for d, v in v_acc_per_dim.items()}

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:03d}] "
              f"train: lm={avg_lm:.4f}  ce={avg_ce:.4f}  orth={avg_orth:.4f}  "
              f"div={avg_div:.4f}  acc={avg_acc:.3f}  | "
              f"val: lm={avg_v_lm:.4f}  ce={avg_v_ce:.4f}  acc={avg_v_acc:.3f}  | "
              f"lr={cur_lr:.2e}  aux_scale={aux_scale:.2f}")


        row = {
            "epoch": epoch, "lr": f"{cur_lr:.8e}", "aux_scale": f"{aux_scale:.4f}",
            "train_lm": f"{avg_lm:.6f}",  "train_ce": f"{avg_ce:.6f}",
            "train_orth": f"{avg_orth:.6f}", "train_div": f"{avg_div:.6f}",
            "train_acc": f"{avg_acc:.4f}",
            "val_lm":   f"{avg_v_lm:.6f}",  "val_ce":  f"{avg_v_ce:.6f}",
            "val_acc":  f"{avg_v_acc:.4f}",
        }
        for d in prism.DIM_NAMES:
            row[f"train_acc_{d}"] = f"{avg_acc_per_dim[d]:.4f}"
        for d in prism.DIM_NAMES:
            row[f"val_acc_{d}"] = f"{avg_v_acc_per_dim[d]:.4f}"
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            w.writerow(row)


        torch.save({"prism": prism.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_lm": avg_v_lm}, last_path)


        if avg_v_lm < best_val:
            best_val = avg_v_lm
            torch.save({"prism": prism.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "val_lm": best_val}, best_path)
            print(f"[BEST] Updated (val_lm={best_val:.4f})")
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] No improvement for {args.early_stop_patience} epochs.")
                break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--feature_dir", type=str, default=FEATURE_DIR)
    parser.add_argument("--report_csv", type=str, default=REPORT_CSV)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--split_json", type=str, default="",
                        help="Participant-grouped development-fold JSON, released with the dataset")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--max_tokens", type=int, default=1024)


    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mamba_lr_mult", type=float, default=3.0,
                        help="LR multiplier for BiMamba parameters")
    parser.add_argument("--weight_decay", type=float, default=2e-2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--aug_noise_std", type=float, default=0.01)
    parser.add_argument("--prefix_norm_lambda", type=float, default=0.0,
                        help="L2 reg on prefix output (default off; final LN should suffice)")


    parser.add_argument("--n_dim_groups", type=int, default=5)
    parser.add_argument("--queries_per_group", type=int, default=6)
    parser.add_argument("--n_free_queries", type=int, default=15,
                        help="0 = pure dim-only, no free channel")
    parser.add_argument("--n_blocks", type=int, default=2,
                        help="number of shared DCA blocks")
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
    parser.add_argument("--use_block_diag_mask", action="store_true", default=True)
    parser.add_argument("--no_block_diag", dest="use_block_diag_mask", action="store_false")


    parser.add_argument("--lam_cls", type=float, default=0.3,
                        help="weight for per-group classification CE")
    parser.add_argument("--lam_orth", type=float, default=0.05,
                        help="weight for query-centroid orthogonality")
    parser.add_argument("--lam_div", type=float, default=0.05,
                        help="weight for attention diversity")
    parser.add_argument("--aux_warmup_epochs", type=int, default=2,
                        help="linear warmup of all three aux losses, 0 to disable")
    parser.add_argument("--lam_route", type=float, default=0.2,
                    help="batch-diversity loss for router; 0 disables")

    args = parser.parse_args()
    if not args.split_json or not Path(args.split_json).is_file():
        parser.error("Supply --split_json; participant-grouped split files will be released with the dataset")
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    save_config_json(args.out_dir, args)


    print(f"Loading model {args.model} ")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(args.device)
    llm_hidden = model.config.hidden_size


    report_map = read_reports(args.report_csv)
    print(f"Loaded {len(report_map)} reports")

    full_dataset = FeatureReportDataset(
        args.feature_dir, report_map, tokenizer, PROMPT_TEMPLATE, args.max_tokens,
        augment=True, aug_noise_std=args.aug_noise_std,
    )
    val_dataset = FeatureReportDataset(
        args.feature_dir, report_map, tokenizer, PROMPT_TEMPLATE, args.max_tokens,
        augment=False, aug_noise_std=0.0,
    )
    print(f"Matched {len(full_dataset)} feature-report pairs")
    if len(full_dataset) == 0:
        raise RuntimeError("No valid features+reports found")

    train_idx, val_idx, split_info = load_grouped_split(full_dataset, args.split_json)
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)
    print(f"Train/Val split: {len(train_subset)}/{len(val_subset)}")

    save_split_json(
        args.out_dir,
        train_ids=[full_dataset.video_ids[i] for i in train_idx],
        val_ids=[full_dataset.video_ids[i] for i in val_idx],
        metadata=split_info,
    )

    sample_path = Path(args.feature_dir) / f"{full_dataset.video_ids[0]}.npy"
    sample_feat = np.load(str(sample_path))
    assert sample_feat.ndim == 2, f"Expected 2D features [S, D], got {sample_feat.ndim}D"
    feat_dim = int(sample_feat.shape[1])


    prism = DimensionAwarePRISM(
        vis_dim=feat_dim,
        hidden=llm_hidden,
        n_dim_groups=args.n_dim_groups,
        queries_per_group=args.queries_per_group,
        n_free_queries=args.n_free_queries,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        dropout=args.dropout,
        ffn_ratio=args.ffn_ratio,
        use_mamba=args.use_mamba,
        mamba_layers=args.mamba_layers,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        bidirectional=args.mamba_bidir,
        use_block_diag_mask=args.use_block_diag_mask,
        dim_names=DEFAULT_DIM_NAMES[:args.n_dim_groups],
    ).to(args.device)

    n_train = sum(p.numel() for p in prism.parameters() if p.requires_grad)
    print(f"[PRISM] vis_dim={feat_dim}  hidden={llm_hidden}  "
          f"prefix_tokens={prism.prefix_tokens}  trainable={n_train/1e6:.2f}M")


    collator = BatchCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=collator)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collator)


    train_loop(prism, model, tokenizer, train_loader, val_loader, args)


if __name__ == "__main__":
    main()
