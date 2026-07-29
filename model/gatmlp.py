import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import coo_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import math

# RDKit & Sklearn
from rdkit import Chem
from rdkit.Chem import rdchem, rdBase, Descriptors, AllChem
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# PyG
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, global_max_pool

# ============================================================================
# 1. 改进的权重初始化
# ============================================================================
def init_weight(m):
    """使用更现代的初始化策略"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# ============================================================================
# 2. 改进的 MPNN 特征提取器
# ============================================================================
class MPNN_FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim=69,
        hidden_dim=32,
        mlp_hidden=128,
        edge_dim=6,
        dropout=0.4,
        num_steps=3,       # message passing 步数
        use_residual=True, # 残差连接
        use_batch_norm=True,
        pooling='mean'
    ):
        super().__init__()
        self.use_residual = use_residual
        self.pooling = pooling
        self.num_steps = num_steps

        # 将边特征映射为 NNConv 需要的卷积核参数
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        # MPNN 核心：NNConv + GRU 更新
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.mpnn = NNConv(hidden_dim, hidden_dim, self.edge_mlp, aggr='mean')
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        self.bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()

        # 残差投影
        if use_residual and input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = None

        # 池化输出维度
        if pooling == 'concat':
            pool_dim = hidden_dim * 3  # mean + max + sum
        else:
            pool_dim = hidden_dim

        # 图级 MLP
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.BatchNorm1d(mlp_hidden // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out_dim = mlp_hidden // 2

    def forward(self, x, edge_index, batch, edge_attr):
        identity = x
        x = self.node_proj(x)
        hidden = x.unsqueeze(0)

        # 多步 message passing
        for _ in range(self.num_steps):
            m = self.mpnn(x, edge_index, edge_attr)
            m = F.relu(m)
            out, hidden = self.gru(m.unsqueeze(0), hidden)
            x = out.squeeze(0)

        x = self.bn(x)

        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            elif identity.shape[-1] != x.shape[-1]:
                identity = self.node_proj(identity)
            x = x + identity

        x = F.relu(x)

        # 图级别池化
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling == 'concat':
            x = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch)
            ], dim=1)

        feat = self.mlp(x)
        return feat

# ============================================================================
# 5. 注意力融合模块
# ============================================================================
class AttentionFusion(nn.Module):
    """使用注意力机制融合多模态特征"""
    def __init__(self, feature_dim, num_modalities=3):
        super().__init__()
        self.num_modalities = num_modalities
        
        # 为每个模态学习查询向量
        self.query = nn.Parameter(torch.randn(1, feature_dim))
        
        # 注意力计算
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1)
        )
        
    def forward(self, features_list):
        """
        features_list: list of [B, D] tensors
        """
        # Stack features: [B, num_modalities, D]
        features = torch.stack(features_list, dim=1)
        
        # 计算注意力权重
        attn_scores = self.attention(features)  # [B, num_modalities, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, num_modalities, 1]
        
        # 加权融合
        fused = (features * attn_weights).sum(dim=1)  # [B, D]
        
        return fused, attn_weights.squeeze(-1)

# ============================================================================
# 6. 改进的融合模型
# ============================================================================
class FusionModel(nn.Module):
    def __init__(
        self,
        mode='fusion',  # 'graph', 'only', 'desc', 'fp'
        aux_dim=0,
        num_classes=2,
        # Graph encoder params
        graph_hidden=32,
        graph_heads=2,
        graph_dropout=0.25,
        # Feature / Fusion params
        feature_dim=128,
        fusion_hidden=128,
        fusion_dropout=0.2,
        use_attention_fusion=False,
        # Regularization
        use_label_smoothing=False,
        label_smoothing=0.1
    ):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.use_attention_fusion = use_attention_fusion
        
        # Graph encoder (MPNN)
        self.graph_encoder = MPNN_FeatureExtractor(
            hidden_dim=graph_hidden,
            dropout=graph_dropout,
            num_steps=3,
            use_residual=True,
            use_batch_norm=True,
            pooling='concat'  # 使用多种池化的组合
        )
        graph_feat_dim = self.graph_encoder.out_dim
        
        if mode == "graph":
            # 仅使用图特征
            self.head = nn.Sequential(
                nn.Linear(graph_feat_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.BatchNorm1d(fusion_hidden // 2),
                nn.ReLU(),
                nn.Dropout(fusion_dropout / 2),
                nn.Linear(fusion_hidden // 2, num_classes),
            )
            self.aux_encoder = None
            self.fusion = None
            self.graph_proj = None
        elif mode == "only":
            # 仅使用 FP 特征
            if aux_dim <= 0:
                raise ValueError("mode='only' 时必须提供有效的 FP 维度 aux_dim。")
            self.head = nn.Sequential(
                nn.Linear(aux_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, num_classes),
            )
            self.aux_encoder = None
            self.fusion = None
            self.graph_proj = None
        else:
            if aux_dim <= 0:
                raise ValueError(f"mode='{mode}' 时必须提供有效的辅助特征维度 aux_dim。")

            # 辅助特征编码器
            self.aux_encoder = nn.Sequential(
                nn.Linear(aux_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(),
            )

            # 特征对齐（将图特征投影到统一维度）
            self.graph_proj = nn.Linear(graph_feat_dim, feature_dim)

            # 融合策略（图 + 指纹编码 + 指纹共性特征）
            if use_attention_fusion:
                self.fusion = AttentionFusion(feature_dim, num_modalities=3)
                fusion_input_dim = feature_dim
            else:
                self.fusion = None
                fusion_input_dim = feature_dim * 3

            # 分类头
            self.head = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.BatchNorm1d(fusion_hidden // 2),
                nn.ReLU(),
                nn.Dropout(fusion_dropout / 2),
                nn.Linear(fusion_hidden // 2, num_classes),
            )
        
        # Label smoothing
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing = label_smoothing
        
        # 初始化权重
        self.apply(init_weight)

    def forward(self, graph, aux_data=None, return_attention=False):
        if self.mode == "only":
            if aux_data is None:
                raise ValueError("mode='only' 需要传入 FP 特征 aux_data。")
            logits = self.head(aux_data)
            return logits if not return_attention else (logits, None)

        # 图特征提取
        g_feat = self.graph_encoder(graph.x, graph.edge_index, graph.batch, graph.edge_attr)

        if self.mode == "graph":
            logits = self.head(g_feat)
            return logits if not return_attention else (logits, None)

        if aux_data is None:
            raise ValueError(f"mode='{self.mode}' 需要传入辅助特征 aux_data。")

        # 指纹辅助编码特征
        a_feat = self.aux_encoder(aux_data)

        # 计算相似度矩阵并提取 common features
        # sim: [B, B], 使用余弦相似度；common_feat: [B, D]
        a_norm = F.normalize(a_feat, p=2, dim=1)
        sim = torch.matmul(a_norm, a_norm.transpose(0, 1))
        sim = F.softmax(sim, dim=1)
        common_feat = torch.matmul(sim, a_feat)

        # 特征对齐
        g_feat_aligned = self.graph_proj(g_feat)

        # 融合（图编码 + 指纹编码 + 共性特征）
        if self.use_attention_fusion:
            combined, attn_weights = self.fusion([g_feat_aligned, a_feat, common_feat])
        else:
            combined = torch.cat([g_feat_aligned, a_feat, common_feat], dim=1)
            attn_weights = None

        # 分类
        logits = self.head(combined)

        return logits if not return_attention else (logits, attn_weights)
    
    def get_loss(self, logits, targets):
        """计算损失（支持 label smoothing）"""
        if self.use_label_smoothing and self.training:
            # Label smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            loss = (-true_dist * log_probs).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(logits, targets)
        return loss
