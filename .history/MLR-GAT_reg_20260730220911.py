"""
MLR-GAT_reg: Multi-Level Representation Graph Attention Network for
HDAC/SIRT Inhibitor Regression (pIC50 / log_value).

This script adapts the MLR-GAT classification architecture to regression:
- Regression head: Linear(1) + MSELoss
- Optuna hyperparameter optimisation (maximise R²)
- 5-fold CV with Deep Ensemble for uncertainty quantification
- Epistemic uncertainty (ensemble std) + Aleatoric uncertainty (sigma_residual)

Framework follows xgb_reg.py; model follows MLR-GAT.py.
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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from dataset_featurizer import FusionDataset, MoleculeFeaturizer
from model.gatmlp import RGCN_FeatureExtractor, AttentionFusion

warnings.filterwarnings("ignore")

# ============================================================================
# Global Settings
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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RUN_MODE = "graph"  # "fusion", "graph", "fp"
OUTPUT_DIR = os.path.join(ROOT_DIR, "result", "mlr_gat_reg", RUN_MODE)
N_TRIALS = 5
OPTUNA_EPOCHS = 60
FINAL_EPOCHS = 120

print(f"Using Device: {DEVICE}")


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Regression Fusion Model
# ============================================================================


class FusionRegModel(nn.Module):
    """MLR-GAT with regression head (Linear → 1).

    Mirrors model.gatmlp.FusionModel but outputs a single scalar.
    """

    def __init__(
        self,
        mode: str = "fusion",
        aux_dim: int = 0,
        graph_hidden: int = 32,
        graph_heads: int = 2,
        graph_dropout: float = 0.25,
        feature_dim: int = 128,
        fusion_hidden: int = 128,
        fusion_dropout: float = 0.2,
        use_attention_fusion: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.use_attention_fusion = use_attention_fusion

        # Graph encoder (RGCN)
        self.graph_encoder = RGCN_FeatureExtractor(
            hidden_dim=graph_hidden,
            dropout=graph_dropout,
            num_layers=3,
            num_relations=4,
            use_residual=True,
            use_batch_norm=True,
            pooling="concat",
        )
        graph_feat_dim = self.graph_encoder.out_dim

        if mode == "graph":
            self.head = nn.Sequential(
                nn.Linear(graph_feat_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.BatchNorm1d(fusion_hidden // 2),
                nn.ReLU(),
                nn.Dropout(fusion_dropout / 2),
                nn.Linear(fusion_hidden // 2, 1),
            )
            self.aux_encoder = None
            self.fusion = None
            self.graph_proj = None
        elif mode == "fp":
            if aux_dim <= 0:
                raise ValueError("mode='fp' requires valid aux_dim.")
            self.head = nn.Sequential(
                nn.Linear(aux_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, 1),
            )
            self.aux_encoder = None
            self.fusion = None
            self.graph_proj = None
        else:
            if aux_dim <= 0:
                raise ValueError(f"mode='{mode}' requires valid aux_dim.")

            self.aux_encoder = nn.Sequential(
                nn.Linear(aux_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(),
            )

            self.graph_proj = nn.Linear(graph_feat_dim, feature_dim)

            if use_attention_fusion:
                self.fusion = AttentionFusion(feature_dim, num_modalities=3)
                fusion_input_dim = feature_dim
            else:
                self.fusion = None
                fusion_input_dim = feature_dim * 3

            self.head = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.BatchNorm1d(fusion_hidden // 2),
                nn.ReLU(),
                nn.Dropout(fusion_dropout / 2),
                nn.Linear(fusion_hidden // 2, 1),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, graph, aux_data=None):
        if self.mode == "fp":
            return self.head(aux_data).squeeze(-1)

        g_feat = self.graph_encoder(
            graph.x, graph.edge_index, graph.batch, graph.edge_attr
        )

        if self.mode == "graph":
            return self.head(g_feat).squeeze(-1)

        a_feat = self.aux_encoder(aux_data)

        a_norm = F.normalize(a_feat, p=2, dim=1)
        sim = torch.matmul(a_norm, a_norm.transpose(0, 1))
        sim = F.softmax(sim, dim=1)
        common_feat = torch.matmul(sim, a_feat)

        g_feat_aligned = self.graph_proj(g_feat)

        if self.use_attention_fusion:
            combined, _ = self.fusion([g_feat_aligned, a_feat, common_feat])
        else:
            combined = torch.cat([g_feat_aligned, a_feat, common_feat], dim=1)

        return self.head(combined).squeeze(-1)


# ============================================================================
# Data Loading
# ============================================================================


class FusionRegDataset(FusionDataset):
    """Regression variant of FusionDataset with log_value labels."""

    def __init__(self, df, featurizer, desc_scaler=None, fit_desc_scaler=False):
        # Temporarily rename the regression column to 'y' for FusionDataset compatibility
        df = df.copy()
        df["y"] = df["log_value"]
        super().__init__(df, featurizer, desc_scaler=desc_scaler, fit_desc_scaler=fit_desc_scaler)


def build_collate_fn():
    def collate_fn(batch):
        graphs = Batch.from_data_list([x["graph"] for x in batch])
        descs = torch.stack([x["desc"] for x in batch])
        fps = torch.stack([x["fp"] for x in batch])
        labels = torch.cat([x["y"] for x in batch], dim=0)  # float32
        return graphs, descs, fps, labels
    return collate_fn


def select_aux_feature(descs, fps, mode: str):
    if mode == "graph":
        return None
    return fps.to(DEVICE)


def make_loader(dataset, collate_fn, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
    )


# ============================================================================
# Training & Evaluation
# ============================================================================


def train_step(model, loader, optimizer, criterion, mode: str, max_grad_norm: float = 5.0):
    model.train()
    total_loss = 0.0

    for graphs, descs, fps, labels in loader:
        graphs = graphs.to(DEVICE)
        labels = labels.to(DEVICE)
        aux = select_aux_feature(descs, fps, mode)

        optimizer.zero_grad()
        preds = model(graphs, aux_data=aux)
        loss = criterion(preds, labels)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def eval_step(model, loader, mode: str):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for graphs, descs, fps, labels in loader:
            graphs = graphs.to(DEVICE)
            aux = select_aux_feature(descs, fps, mode)
            preds = model(graphs, aux_data=aux)
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())

    preds = torch.cat(preds_all, dim=0).numpy()
    labels = torch.cat(labels_all, dim=0).numpy()

    rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    try:
        r2 = float(r2_score(labels, preds))
    except ValueError:
        r2 = float("-inf")

    return r2, rmse, labels, preds


def train_single_run(
    model, train_loader, valid_loader,
    epochs: int, lr: float, weight_decay: float,
    early_stop_patience: int, scheduler_patience: int,
    mode: str, scheduler_factor: float = 0.5,
):
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=1e-5,
    )
    criterion = nn.MSELoss()

    best_r2 = float("-inf")
    best_rmse = 0.0
    best_state = None
    no_improve = 0

    history = {"loss": [], "val_r2": [], "val_rmse": []}

    for _ in range(epochs):
        train_loss = train_step(model, train_loader, optimizer, criterion, mode=mode)
        val_r2, val_rmse, _, _ = eval_step(model, valid_loader, mode=mode)
        scheduler.step(val_r2)

        history["loss"].append(train_loss)
        history["val_r2"].append(val_r2)
        history["val_rmse"].append(val_rmse)

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            break

    return best_r2, best_rmse, best_state, history


# ============================================================================
# Optuna Objective
# ============================================================================


def objective(trial, train_dataset, valid_dataset, collate_fn, fp_dim, mode: str):
    params = {
        "graph_hidden": trial.suggest_categorical("graph_hidden", [16, 32, 48, 64]),
        "graph_heads": trial.suggest_int("graph_heads", 1, 4),
        "graph_dropout": trial.suggest_float("graph_dropout", 0.10, 0.45),
        "feature_dim": trial.suggest_categorical("feature_dim", [64, 128, 256]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [64, 128, 256]),
        "fusion_dropout": trial.suggest_float("fusion_dropout", 0.05, 0.40),
        "use_attention_fusion": trial.suggest_categorical("use_attention_fusion", [False, True]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 4, 10),
    }

    train_loader = make_loader(train_dataset, collate_fn, batch_size=params["batch_size"], shuffle=True)
    valid_loader = make_loader(valid_dataset, collate_fn, batch_size=params["batch_size"], shuffle=False)

    aux_dim = fp_dim if mode in {"fusion", "fp"} else 0

    model = FusionRegModel(
        mode=mode,
        aux_dim=aux_dim,
        graph_hidden=params["graph_hidden"],
        graph_heads=params["graph_heads"],
        graph_dropout=params["graph_dropout"],
        feature_dim=params["feature_dim"],
        fusion_hidden=params["fusion_hidden"],
        fusion_dropout=params["fusion_dropout"],
        use_attention_fusion=params["use_attention_fusion"],
    ).to(DEVICE)

    best_r2, _, _, _ = train_single_run(
        model=model, train_loader=train_loader, valid_loader=valid_loader,
        epochs=OPTUNA_EPOCHS, lr=params["lr"],
        weight_decay=params["weight_decay"],
        early_stop_patience=params["early_stop_patience"],
        scheduler_patience=6, mode=mode,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_r2 if best_r2 != float("-inf") else -1e10


# ============================================================================
# Plotting
# ============================================================================


def _plot_results(target_name, y_true, y_pred, y_std, r2, rmse):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 1) True vs Predicted with uncertainty error bars
    axes[0].errorbar(
        y_true, y_pred, yerr=y_std,
        fmt="o", alpha=0.55, ecolor="gray", elinewidth=0.8, capsize=2,
    )
    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot([min_v, max_v], [min_v, max_v], "r--", lw=2)
    axes[0].set_xlabel("Actual log_value")
    axes[0].set_ylabel("Predicted log_value (OOF)")
    axes[0].set_title(f"Actual vs Predicted (R²={r2:.3f}, RMSE={rmse:.3f})")

    # 2) Residual distribution
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color="steelblue")
    axes[1].axvline(0, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("Residual (Actual - OOF Predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution (OOF Ensemble Mean)")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"{target_name}_mlr_gat_reg.png")
    fig.savefig(fig_path)
    plt.close(fig)


# ============================================================================
# Core Pipeline
# ============================================================================


def make_mlr_gat_regression(target_name, mode="fusion", ensemble_n=5, std_threshold=0.3):
    data_path = os.path.join(DATA_DIR, target_name)
    data = pd.read_csv(os.path.join(data_path, "reg_origin.csv"), sep="\t")

    if len(data) < 5:
        print(f"[{target_name}] Too few samples ({len(data)}), skipping.")
        return {
            "R2": None, "R2_std": None, "RMSE": None, "RMSE_std": None,
            "best_params": None,
            "mean_uncertainty_std": None,
            f"high_uncertainty_ratio_std_gt_{std_threshold}": None,
            "sigma_residual_train": None, "ensemble_n": None,
            "fold_R2": None, "fold_RMSE": None,
        }

    # --- Build dataset ---
    featurizer = MoleculeFeaturizer()
    desc_scaler = StandardScaler()
    dataset = FusionRegDataset(data, featurizer, desc_scaler=desc_scaler, fit_desc_scaler=True)
    y_all = data["log_value"].values
    smiles_all = data["smiles"].values

    fp_dim = dataset[0]["fp"].shape[0]
    collate_fn = build_collate_fn()

    # --- Optuna hyperparameter search ---
    print(f"\n[{target_name}] Starting Optuna (mode={mode}, n_trials={N_TRIALS})")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx_0, valid_idx_0 = next(kf.split(data))
    train_subset = torch.utils.data.Subset(dataset, train_idx_0)
    valid_subset = torch.utils.data.Subset(dataset, valid_idx_0)

    study.optimize(
        lambda trial: objective(trial, train_subset, valid_subset, collate_fn, fp_dim, mode),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"[{target_name}] Best val R²: {study.best_value:.4f}")
    print(f"[{target_name}] Best params: {study.best_params}")

    # --- 5-Fold CV OOF prediction (ensemble) ---
    best_params = dict(study.best_params)
    batch_size = best_params.pop("batch_size")
    early_stop_patience = best_params.pop("early_stop_patience")
    lr = best_params.pop("lr")
    weight_decay = best_params.pop("weight_decay")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_mean = np.full(len(data), np.nan)
    oof_std = np.full(len(data), np.nan)
    fold_r2_list = []
    fold_rmse_list = []

    aux_dim = fp_dim if mode in {"fusion", "fp"} else 0

    print(f"[{target_name}] Starting 5-Fold CV OOF (ensemble_n={ensemble_n})")

    for fold_i, (train_idx, valid_idx) in enumerate(kf.split(data)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        valid_subset = torch.utils.data.Subset(dataset, valid_idx)

        train_loader = make_loader(train_subset, collate_fn, batch_size, shuffle=True)
        valid_loader = make_loader(valid_subset, collate_fn, batch_size, shuffle=False)

        fold_preds = []
        for seed_off in range(ensemble_n):
            seed_everything(SEED + seed_off)
            model = FusionRegModel(mode=mode, aux_dim=aux_dim, **best_params).to(DEVICE)
            _, _, best_state, _ = train_single_run(
                model=model, train_loader=train_loader, valid_loader=valid_loader,
                epochs=FINAL_EPOCHS, lr=lr, weight_decay=weight_decay,
                early_stop_patience=early_stop_patience, scheduler_patience=8,
                mode=mode,
            )
            if best_state is not None:
                model.load_state_dict(best_state)
            _, _, y_val, y_pred_val = eval_step(model, valid_loader, mode=mode)
            fold_preds.append(y_pred_val)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pred_mat = np.vstack(fold_preds)
        fold_mean = pred_mat.mean(axis=0)
        fold_std = pred_mat.std(axis=0)
        oof_mean[valid_idx] = fold_mean
        oof_std[valid_idx] = fold_std

        y_val_true = y_all[valid_idx]
        fold_r2_list.append(r2_score(y_val_true, fold_mean))
        fold_rmse_list.append(np.sqrt(mean_squared_error(y_val_true, fold_mean)))

        print(f"  Fold {fold_i+1} R²={fold_r2_list[-1]:.3f}, RMSE={fold_rmse_list[-1]:.3f}")

    # --- Full-data training for final ensemble (for saving) ---
    seed_everything(SEED)
    full_loader = make_loader(dataset, collate_fn, batch_size, shuffle=True)

    final_models = []
    final_train_preds = []
    for seed_off in range(ensemble_n):
        seed_everything(SEED + seed_off)
        model = FusionRegModel(mode=mode, aux_dim=aux_dim, **best_params).to(DEVICE)
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        criterion = nn.MSELoss()
        for _ in range(FINAL_EPOCHS):
            train_step(model, full_loader, optimizer, criterion, mode=mode)
        final_models.append(model)
        _, _, _, preds_all = eval_step(model, full_loader, mode=mode)
        final_train_preds.append(preds_all)

    train_pred_mat = np.vstack(final_train_preds)
    train_mean_pred = train_pred_mat.mean(axis=0)
    sigma_res = float(np.std(y_all - train_mean_pred))

    # --- Metrics ---
    r2 = r2_score(y_all, oof_mean)
    rmse = np.sqrt(mean_squared_error(y_all, oof_mean))

    mean_uncertainty_std = float(np.mean(oof_std))
    high_uncertainty_ratio = float(np.mean(oof_std > std_threshold))

    total_sigma = np.sqrt(oof_std ** 2 + sigma_res ** 2)
    pi_low = oof_mean - 1.96 * total_sigma
    pi_high = oof_mean + 1.96 * total_sigma

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save models
    torch.save(
        [m.state_dict() for m in final_models],
        os.path.join(OUTPUT_DIR, f"{target_name}_mlr_gat_reg_ensemble.pt"),
    )

    # Save uncertainty parameters
    with open(
        os.path.join(OUTPUT_DIR, f"{target_name}_mlr_gat_reg_params.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({
            "sigma_res": sigma_res,
            "best_params": study.best_params,
            "ensemble_n": ensemble_n,
            "mode": mode,
        }, f, ensure_ascii=False, indent=2)

    # Save OOF per-sample predictions
    pred_df = pd.DataFrame({
        "smiles": smiles_all,
        "actual_log_value": y_all,
        "pred_mean": oof_mean,
        "pred_std": oof_std,
        "pi_low_95": pi_low,
        "pi_high_95": pi_high,
    })
    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{target_name}_mlr_gat_reg_test_with_uncertainty.csv"),
        index=False,
    )

    # Plot
    _plot_results(target_name, y_all, oof_mean, oof_std, r2, rmse)

    metrics = {
        "R2": float(r2),
        "R2_std": float(np.std(fold_r2_list)),
        "RMSE": float(rmse),
        "RMSE_std": float(np.std(fold_rmse_list)),
        "best_params": study.best_params,
        "mean_uncertainty_std": mean_uncertainty_std,
        f"high_uncertainty_ratio_std_gt_{std_threshold}": high_uncertainty_ratio,
        "sigma_residual_train": sigma_res,
        "ensemble_n": ensemble_n,
        "fold_R2": fold_r2_list,
        "fold_RMSE": fold_rmse_list,
    }
    return metrics


# ============================================================================
# Main
# ============================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"MLR-GAT Regression — Deep Ensemble with Uncertainty (mode={RUN_MODE})")

    target_names = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )

    if not target_names:
        print("No target directories found under data/. Exiting.")
        return

    print(f"Found {len(target_names)} target(s): {target_names}")
    print("=" * 60)

    target_col, method_col, value_col = [], [], []

    for target_name in target_names:
        reg_path = os.path.join(DATA_DIR, target_name, "reg_origin.csv")
        if not os.path.exists(reg_path):
            print(f"[{target_name}] No reg_origin.csv, skipping.")
            continue

        print(f"\n>>> Processing {target_name} ...")
        seed_everything(SEED)
        try:
            metrics = make_mlr_gat_regression(target_name, mode=RUN_MODE, ensemble_n=5, std_threshold=0.3)
            for k, v in metrics.items():
                target_col.append(target_name)
                method_col.append(k)
                value_col.append(v)
        except Exception as e:
            import traceback
            print(f"  Failed: {e}")
            traceback.print_exc()
            continue

    summary_df = pd.DataFrame({
        "target": target_col,
        "method": method_col,
        "getvalue": value_col,
    })
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "mlr_gat_reg_summary.csv"), index=False)
    print(f"\nDone. Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
