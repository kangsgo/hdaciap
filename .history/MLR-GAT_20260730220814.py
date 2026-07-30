"""
MLR-GAT: Multi-Level Representation Graph Attention Network for
HDAC/SIRT Inhibitor Classification.

This script trains a GNN+FP fusion model with Optuna hyperparameter
optimization, scaffold-based train/validation splits, and test-set
evaluation on ECFP fingerprints and molecular graphs.

ACS JCIM Publication Style
"""

import json
import os
import random
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from dataset_featurizer import FusionDataset, MoleculeFeaturizer
from model.gatmlp import FusionModel
from scaffold_split import light_scaffold_split

warnings.filterwarnings("ignore")

# ============================================================================
# Global Settings — ACS JCIM Publication Formatting
# ============================================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
DATA_DIR = "data"
# --- 运行模式选择 ---
# 可选: "fusion" (图+指纹), "graph" (仅图), "fp" (仅指纹)
RUN_MODE = "fusion"
# 根据模式自动切换输出目录
OUTPUT_DIR = f"result/mlr_gat/{RUN_MODE}"
N_TRIALS = 30
OPTUNA_EPOCHS = 40
FINAL_EPOCHS = 60

print(f"Using Device: {DEVICE}")


def seed_everything(seed: int = SEED):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Data Loading Utilities
# ============================================================================


def build_collate_fn():
    """Build a collate function that batches graphs, descriptors, FPs, and labels."""

    def collate_fn(batch):
        graphs = Batch.from_data_list([x["graph"] for x in batch])
        descs = torch.stack([x["desc"] for x in batch])
        fps = torch.stack([x["fp"] for x in batch])
        # CrossEntropyLoss 需要 Long 类型标签，FusionDataset 默认存 float32
        labels = torch.cat([x["y"] for x in batch], dim=0).long()
        return graphs, descs, fps, labels

    return collate_fn


def select_aux_feature(descs, fps, mode: str):
    """Select auxiliary features based on training mode."""
    if mode == "graph":
        return None
    # fusion, only, desc, fp all use FP as auxiliary
    return fps.to(DEVICE)


def make_loader(dataset, collate_fn, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and ensure required fields exist."""
    df = df.copy()
    df.rename(columns=str.lower, inplace=True)
    required = {"smiles", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


# ============================================================================
# Training & Evaluation
# ============================================================================


def train_step(model, loader, optimizer, criterion, mode: str, max_grad_norm: float = 5.0):
    """Single training epoch."""
    model.train()
    total_loss = 0.0

    for graphs, descs, fps, labels in loader:
        graphs = graphs.to(DEVICE)
        labels = labels.to(DEVICE)
        aux = select_aux_feature(descs, fps, mode)

        optimizer.zero_grad()
        logits = model(graphs, aux_data=aux)
        loss = criterion(logits, labels)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def eval_step(model, loader, mode: str):
    """Evaluate model on a given data loader."""
    model.eval()
    logits_all, labels_all = [], []

    with torch.no_grad():
        for graphs, descs, fps, labels in loader:
            graphs = graphs.to(DEVICE)
            aux = select_aux_feature(descs, fps, mode)
            logits = model(graphs, aux_data=aux)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    preds = torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()

    labels_np = labels.numpy()
    preds_np = preds.numpy()

    acc = accuracy_score(labels_np, preds_np)
    bal_acc = balanced_accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average="binary", zero_division=0)

    try:
        auc_val = roc_auc_score(labels_np, probs)
    except ValueError:
        auc_val = 0.0

    return acc, bal_acc, f1, auc_val, labels_np, preds_np, probs


def train_single_run(
    model,
    train_loader,
    valid_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stop_patience: int,
    scheduler_patience: int,
    run_mode: str,
    scheduler_factor: float = 0.5,
):
    """Train the model for a given number of epochs with early stopping.

    Returns
    -------
    tuple
        (best_val_auc, best_val_acc, best_val_f1, best_state_dict, history_dict)
    """
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-5,
    )
    criterion = nn.CrossEntropyLoss()

    best_auc = -1.0
    best_acc = 0.0
    best_bal_acc = 0.0
    best_f1 = 0.0
    best_state = None
    no_improve = 0

    history = {"loss": [], "val_acc": [], "val_bal_acc": [], "val_auc": [], "val_f1": []}

    for _ in range(epochs):
        train_loss = train_step(model, train_loader, optimizer, criterion, mode=run_mode)
        val_acc, val_bal_acc, val_f1, val_auc, _, _, _ = eval_step(
            model, valid_loader, mode=run_mode
        )
        scheduler.step(val_auc)

        history["loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            best_bal_acc = val_bal_acc
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            break

    return best_auc, best_acc, best_bal_acc, best_f1, best_state, history


# ============================================================================
# Optuna Objective
# ============================================================================


def objective(trial, train_dataset, valid_dataset, collate_fn, fp_dim, run_mode: str):
    """Optuna objective: maximize validation AUC."""
    params = {
        "graph_hidden": trial.suggest_categorical("graph_hidden", [16, 32, 48, 64]),
        "graph_heads": trial.suggest_int("graph_heads", 1, 4),
        "graph_dropout": trial.suggest_float("graph_dropout", 0.10, 0.45),
        "feature_dim": trial.suggest_categorical("feature_dim", [64, 128, 256]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [64, 128, 256]),
        "fusion_dropout": trial.suggest_float("fusion_dropout", 0.05, 0.40),
        "use_attention_fusion": trial.suggest_categorical(
            "use_attention_fusion", [False, True]
        ),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 4, 10),
    }

    train_loader = make_loader(
        train_dataset, collate_fn, batch_size=params["batch_size"], shuffle=True
    )
    valid_loader = make_loader(
        valid_dataset, collate_fn, batch_size=params["batch_size"], shuffle=False
    )

    aux_dim = fp_dim if run_mode in {"fusion", "fp", "desc"} else 0

    model = FusionModel(
        mode=run_mode,
        aux_dim=aux_dim,
        graph_hidden=params["graph_hidden"],
        graph_heads=params["graph_heads"],
        graph_dropout=params["graph_dropout"],
        feature_dim=params["feature_dim"],
        fusion_hidden=params["fusion_hidden"],
        fusion_dropout=params["fusion_dropout"],
        use_attention_fusion=params["use_attention_fusion"],
    ).to(DEVICE)

    best_auc, best_acc, best_bal_acc, best_f1, best_state, _ = train_single_run(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=OPTUNA_EPOCHS,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        early_stop_patience=params["early_stop_patience"],
        scheduler_patience=6,
        run_mode=run_mode,
        scheduler_factor=0.5,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_f1


# ============================================================================
# Plotting
# ============================================================================


def _plot_results(target_name, y_test, y_pred, y_proba):
    """Generate and save a combined confusion-matrix / ROC figure.

    Parameters
    ----------
    target_name : str
    y_test : array-like
    y_pred : array-like
    y_proba : array-like
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Inactive", "Active"],
        yticklabels=["Inactive", "Active"],
        cbar=False,
        annot_kws={"fontsize": 8},
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{target_name} — Confusion Matrix")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(
        fpr, tpr, color="darkorange", lw=1.5,
        label=f"AUC = {roc_auc:.3f}"
    )
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1.0, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{target_name} — ROC Curve")
    axes[1].legend(loc="lower right", frameon=False)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"{target_name}_mlr_gat_results.png")
    fig.savefig(fig_path)
    plt.close(fig)


# ============================================================================
# Core Pipeline
# ============================================================================


def train_and_evaluate(target_name, run_mode="fusion"):
    """Run the full MLR-GAT pipeline for a single target.

    Parameters
    ----------
    target_name : str
        Name of the target subdirectory (e.g., 'HDAC1', 'SIRT1').
    run_mode : str
        Training mode: 'fusion', 'graph', or 'fp'.

    Returns
    -------
    dict
        Dictionary with validation and test metrics plus best parameters.
    """
    data_path = os.path.join(DATA_DIR, target_name)

    # --- Load data ---
    train_valid_df = pd.read_csv(os.path.join(data_path, "train_classic.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test_classic.csv"))

    train_valid_df = normalize_columns(train_valid_df)
    test_df = normalize_columns(test_df)

    train_df, valid_df = light_scaffold_split(train_valid_df, valid_size=0.2)

    # --- Class balancing (training set only) ---
    train_df, _ = RandomOverSampler(random_state=SEED).fit_resample(
        train_df, train_df["y"]
    )

    # --- Build datasets ---
    featurizer = MoleculeFeaturizer()
    desc_scaler = StandardScaler()

    train_dataset = FusionDataset(
        train_df, featurizer, desc_scaler=desc_scaler, fit_desc_scaler=True
    )
    valid_dataset = FusionDataset(
        valid_df, featurizer, desc_scaler=desc_scaler, fit_desc_scaler=False
    )
    test_dataset = FusionDataset(
        test_df, featurizer, desc_scaler=desc_scaler, fit_desc_scaler=False
    )

    fp_dim = train_dataset[0]["fp"].shape[0]
    collate_fn = build_collate_fn()

    # --- Optuna hyperparameter search ---
    print(f"\n[{target_name}] Starting Optuna (mode={run_mode}, "
          f"n_trials={N_TRIALS}, epochs/trial={OPTUNA_EPOCHS})")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective(
            trial,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            collate_fn=collate_fn,
            fp_dim=fp_dim,
            run_mode=run_mode,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"[{target_name}] Best val F1: {study.best_value:.4f}")
    print(f"[{target_name}] Best params: {study.best_params}")

    # --- Retrain with best parameters ---
    best_params = dict(study.best_params)
    batch_size = best_params.pop("batch_size")
    early_stop_patience = best_params.pop("early_stop_patience")
    lr = best_params.pop("lr")
    weight_decay = best_params.pop("weight_decay")

    train_loader = make_loader(train_dataset, collate_fn, batch_size, shuffle=True)
    valid_loader = make_loader(valid_dataset, collate_fn, batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, collate_fn, batch_size, shuffle=False)

    aux_dim = fp_dim if run_mode in {"fusion", "fp", "desc"} else 0
    best_model = FusionModel(mode=run_mode, aux_dim=aux_dim, **best_params).to(DEVICE)

    best_val_auc, best_val_acc, best_val_bal_acc, best_val_f1, best_state, _ = train_single_run(
        model=best_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=FINAL_EPOCHS,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_patience=early_stop_patience,
        scheduler_patience=8,
        run_mode=run_mode,
        scheduler_factor=0.5,
    )

    if best_state is not None:
        best_model.load_state_dict(best_state)

    # --- Test-set evaluation ---
    test_acc, test_bal_acc, test_f1, test_auc, labels_np, preds_np, probs = eval_step(
        best_model, test_loader, mode=run_mode
    )

    print(f"\n[{target_name}] Results:")
    print(f"  Best Val ACC : {best_val_acc:.4f}")
    print(f"  Best Val BACC: {best_val_bal_acc:.4f}")
    print(f"  Best Val F1  : {best_val_f1:.4f}")
    print(f"  Best Val AUC : {best_val_auc:.4f}")
    print(f"  Test ACC     : {test_acc:.4f}")
    print(f"  Test BACC    : {test_bal_acc:.4f}")
    print(f"  Test F1      : {test_f1:.4f}")
    print(f"  Test AUC     : {test_auc:.4f}")

    # --- Save artifacts ---
    torch.save(
        best_model.state_dict(),
        os.path.join(OUTPUT_DIR, f"{target_name}_mlr_gat_model.pt"),
    )
    with open(
        os.path.join(OUTPUT_DIR, f"{target_name}_best_params.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    # --- Publication-quality figure ---
    _plot_results(target_name, labels_np, preds_np, probs)

    return {
        "best_val_acc": best_val_acc,
        "best_val_bal_acc": best_val_bal_acc,
        "best_val_f1": best_val_f1,
        "best_val_auc": best_val_auc,
        "test_acc": test_acc,
        "test_bal_acc": test_bal_acc,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "best_params": study.best_params,
    }


# ============================================================================
# Main Execution
# ============================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_names = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )

    if not target_names:
        print("No target directories found under data/. Exiting.")
        return

    print(f"Found {len(target_names)} target(s): {target_names}")
    print("=" * 60)

    records = []
    for name in target_names:
        print(f"\n>>> Processing {name} ...")
        seed_everything(SEED)
        try:
            metrics = train_and_evaluate(name, run_mode=RUN_MODE)
            records.append((name, "best_val_acc", metrics["best_val_acc"]))
            records.append((name, "best_val_bal_acc", metrics["best_val_bal_acc"]))
            records.append((name, "best_val_f1", metrics["best_val_f1"]))
            records.append((name, "best_val_auc", metrics["best_val_auc"]))
            records.append((name, "test_acc", metrics["test_acc"]))
            records.append((name, "test_bal_acc", metrics["test_bal_acc"]))
            records.append((name, "test_f1", metrics["test_f1"]))
            records.append((name, "test_auc", metrics["test_auc"]))
            records.append((name, "best_params", str(metrics["best_params"])))
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # --- Save summary ---
    df_results = pd.DataFrame(records, columns=["target", "metric", "value"])
    csv_path = os.path.join(OUTPUT_DIR, "mlr_gat_summary.csv")
    df_results.to_csv(csv_path, index=False)

    print(f"\nDone. Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
