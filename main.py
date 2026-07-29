"""多模态分子分类训练入口（Optuna 版，聚焦 GNN+FP）。

主要功能：
1. 加载并预处理数据；
2. 构建 DataLoader（描述符 scaler 仅在训练集拟合）；
3. 使用 Optuna 对 GNN+FP 模型超参数搜索（目标：验证集 AUC）；
4. 使用最佳参数重训并在测试集评估；
5. 保存最佳模型与最优参数。
"""

import argparse
import json
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from dataset_featurizer import FusionDataset, MoleculeFeaturizer
from model.gatmlp import FusionModel
from scaffold_split import light_scaffold_split

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import seaborn as sns



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
print(f"Using Device: {DEVICE}")


def seed_everything(seed: int = 42):
    """固定随机种子，尽可能保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tune for molecular classifier")
    parser.add_argument("--n_trials", type=int, default=20, help="Optuna 试验次数")
    parser.add_argument(
        "--optuna_epochs", type=int, default=40, help="每个 trial 的最大训练轮数"
    )
    parser.add_argument(
        "--final_epochs", type=int, default=120, help="最佳参数重训时最大训练轮数"
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="fusion",
        choices=["fusion", "only", "graph"],
        help="训练模式：fusion(图+FP) / only(仅FP) / graph(仅图)",
    )
    return parser.parse_args()


def build_collate_fn():
    """构建用于 DataLoader 的 batch 拼接函数。"""

    def collate_fn(batch):
        graphs = Batch.from_data_list([x["graph"] for x in batch])
        descs = torch.stack([x["desc"] for x in batch])
        fps = torch.stack([x["fp"] for x in batch])
        labels = torch.cat([x["y"] for x in batch], dim=0)
        return graphs, descs, fps, labels

    return collate_fn


def select_aux_feature(descs, fps, mode: str):
    """根据训练模式选择辅助特征。"""
    if mode == "graph":
        return None
    if mode == "only":
        return fps.to(DEVICE)
    if mode == "desc":
        return descs.to(DEVICE)
    if mode == "fp":
        return fps.to(DEVICE)
    # fusion 默认使用 FP 作为辅助特征
    return fps.to(DEVICE)


def train_step(
    model,
    loader,
    optimizer,
    criterion,
    mode: str,
    max_grad_norm: float = 5.0,
):
    """执行一个 epoch 的训练。"""
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
    """在验证/测试集上评估模型并返回指标。"""
    model.eval()
    logits_all, labels_all = [], []

    with torch.no_grad():
        for graphs, descs, fps, labels in loader:
            graphs = graphs.to(DEVICE)
            aux = select_aux_feature(descs, fps, mode)
            logits = model(graphs, aux_data=aux)

            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())

    logits = torch.cat(logits_all, dim=0)  # [N, num_classes]
    labels = torch.cat(labels_all, dim=0)  # [N]
    preds = torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()

    labels_np = labels.numpy()
    preds_np = preds.numpy()

    acc = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(labels_np, probs)
    except Exception:
        auc = 0.0

    return acc, f1, auc, labels_np, preds_np, probs


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名格式并确保训练逻辑所需字段存在。"""
    df = df.copy()
    df.rename(columns=str.lower, inplace=True)
    df.rename(columns={"smiles": "smiles"}, inplace=True)

    required_cols = {"smiles", "y"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入数据缺少必要字段: {missing_cols}")
    return df


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
    """单次训练流程，返回最佳验证 AUC 与最佳权重。"""
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
    best_f1 = 0.0
    best_state = None
    no_improve = 0

    history = {"loss": [], "val_acc": [], "val_auc": [], "val_f1": []}

    for _ in range(epochs):
        train_loss = train_step(model, train_loader, optimizer, criterion, mode=run_mode)
        val_acc, val_f1, val_auc, _, _, _ = eval_step(
            model, valid_loader, mode=run_mode
        )

        scheduler.step(val_auc)

        history["loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            break

    return best_auc, best_acc, best_f1, best_state, history


def make_loader(dataset, collate_fn, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def objective(
    trial, train_dataset, valid_dataset, collate_fn, fp_dim, optuna_epochs, run_mode: str
):
    """Optuna 目标函数：最大化验证集 AUC。"""
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

    aux_dim = fp_dim if run_mode in {"fusion", "only", "desc", "fp"} else 0

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

    best_auc, _, _, _, _ = train_single_run(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=optuna_epochs,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        early_stop_patience=params["early_stop_patience"],
        scheduler_patience=6,
        run_mode=run_mode,
        scheduler_factor=0.5,
    )

    # 显存清理，减少多 trial 累积占用
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_auc


def find_data_directories(root_dir: str = "."):
    """遍历目录，筛选同时包含 classic_train.csv 和 classic_test.csv 的目录。"""
    matched = []
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        file_set = set(files)
        if {"train_classic.csv", "test_classic.csv"}.issubset(file_set):
            matched.append(current_root)
    return sorted(matched)


def save_metric_report(
    target_dir: str,
    best_val_acc: float,
    best_val_f1: float,
    best_val_auc: float,
    test_acc: float,
    test_f1: float,
    test_auc: float,
):
    """将关键指标写入目录内文本文件，便于后续查阅。"""
    report_path = os.path.join(target_dir, "classic_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"   - Best Val ACC: {best_val_acc:.4f}\n")
        f.write(f"   - Best Val F1 : {best_val_f1:.4f}\n")
        f.write(f"   - Best Val AUC: {best_val_auc:.4f}\n")
        f.write(f"   - Test ACC    : {test_acc:.4f}\n")
        f.write(f"   - Test F1     : {test_f1:.4f}\n")
        f.write(f"   - Test AUC    : {test_auc:.4f}\n")


def run_single_directory(data_dir: str, args):
    """在单个目录上执行完整训练、评估与导出流程。"""
    print(f"\n{'=' * 80}")
    print(f"📂 处理目录: {data_dir}")
    print(f"{'=' * 80}")

    train_path = os.path.join(data_dir, "train_classic.csv")
    test_path = os.path.join(data_dir, "test_classic.csv")

    # 1) 数据加载
    train_valid_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_valid_df = normalize_columns(train_valid_df)
    test_df = normalize_columns(test_df)

    train_df, valid_df = light_scaffold_split(train_valid_df, valid_size=0.2)

    # 2) 类别重采样（仅训练集）
    train_df, _ = RandomOverSampler().fit_resample(train_df, train_df["y"])

    # 3) 构建数据集（描述符标准化仅在训练集 fit）
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

    sample = train_dataset[0]
    fp_dim = sample["fp"].shape[0]
    print(f"\nFingerprint dim: {fp_dim}")

    collate_fn = build_collate_fn()

    # 4) Optuna 搜索
    print(
        f"\n🔍 开始 Optuna 调参（mode={args.run_mode}, n_trials={args.n_trials}, epochs/trial={args.optuna_epochs}）..."
    )
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
            optuna_epochs=args.optuna_epochs,
            run_mode=args.run_mode,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\n✅ Optuna 完成")
    print(f"Best trial value (Val AUC): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # 5) 用最佳参数重训
    best_params = dict(study.best_params)
    batch_size = best_params.pop("batch_size")
    early_stop_patience = best_params.pop("early_stop_patience")
    lr = best_params.pop("lr")
    weight_decay = best_params.pop("weight_decay")

    train_loader = make_loader(
        train_dataset, collate_fn, batch_size=batch_size, shuffle=True
    )
    valid_loader = make_loader(
        valid_dataset, collate_fn, batch_size=batch_size, shuffle=False
    )
    test_loader = make_loader(
        test_dataset, collate_fn, batch_size=batch_size, shuffle=False
    )

    aux_dim = fp_dim if args.run_mode in {"fusion", "only", "desc", "fp"} else 0
    best_model = FusionModel(mode=args.run_mode, aux_dim=aux_dim, **best_params).to(
        DEVICE
    )

    best_val_auc, best_val_acc, best_val_f1, best_state, _ = train_single_run(
        model=best_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.final_epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_patience=early_stop_patience,
        scheduler_patience=8,
        run_mode=args.run_mode,
        scheduler_factor=0.5,
    )

    if best_state is not None:
        best_model.load_state_dict(best_state)

    # 6) 测试评估
    test_acc, test_f1, test_auc, labels_np, preds_np, probs = eval_step(
        best_model, test_loader, mode=args.run_mode
    )

    print("\n🏆 最优 GNN+FP 结果")
    print(f"   - Best Val ACC: {best_val_acc:.4f}")
    print(f"   - Best Val F1 : {best_val_f1:.4f}")
    print(f"   - Best Val AUC: {best_val_auc:.4f}")
    print(f"   - Test ACC    : {test_acc:.4f}")
    print(f"   - Test F1     : {test_f1:.4f}")
    print(f"   - Test AUC    : {test_auc:.4f}")

    # 7) 保存模型、参数与指标
    model_path = os.path.join(data_dir, "Best_Modal_Classifier_Optuna.pt")
    params_path = os.path.join(data_dir, "optuna_best_params.json")
    fig_path = os.path.join(data_dir, "classic_mu.png")

    torch.save(best_model.state_dict(), model_path)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    save_metric_report(
        target_dir=data_dir,
        best_val_acc=best_val_acc,
        best_val_f1=best_val_f1,
        best_val_auc=best_val_auc,
        test_acc=test_acc,
        test_f1=test_f1,
        test_auc=test_auc,
    )

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    cm = confusion_matrix(labels_np, preds_np)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Inactive", "Active"],
        yticklabels=["Inactive", "Active"],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix (Test Set)")

    # ROC 曲线
    fpr, tpr, _ = roc_curve(labels_np, probs)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    print(
        f"\n📁 已保存：{model_path} / {params_path} / {fig_path} / {os.path.join(data_dir, 'classic_metrics.txt')}"
    )


def main():
    args = parse_args()

    data_dirs = find_data_directories(".")
    if not data_dirs:
        print("未找到包含 classic_train.csv 和 classic_test.csv 的目录，任务结束。")
        return

    print(f"共发现 {len(data_dirs)} 个可处理目录。")
    for idx, data_dir in enumerate(data_dirs, start=1):
        print(f"\n>>> [{idx}/{len(data_dirs)}] 即将处理: {data_dir}")
        seed_everything(SEED)
        try:
            run_single_directory(data_dir, args)
        except Exception as e:
            print(f"❌ 目录处理失败: {data_dir}")
            print(f"   原因: {e}")
            continue


if __name__ == "__main__":
    main()
