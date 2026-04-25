import os
import time
import random
import threading
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import sparse
from sklearn.metrics import mean_squared_error, ndcg_score
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv as PYG_GCNConv, GATConv, SAGEConv as PYG_SAGEConv, SGConv

try:
    import psutil
except Exception:
    psutil = None


class PeakMemoryTracker:
    """Track process RSS peak during a specific training window."""
    def __init__(self, interval_sec=0.05):
        self.interval_sec = interval_sec
        self._running = False
        self._thread = None
        self._peak_mb = 0.0

    def _sample_loop(self):
        process = psutil.Process(os.getpid()) if psutil is not None else None
        while self._running:
            if process is not None:
                rss_mb = process.memory_info().rss / 1024 / 1024
                if rss_mb > self._peak_mb:
                    self._peak_mb = rss_mb
            time.sleep(self.interval_sec)

    def start(self):
        if psutil is None or self._running:
            return
        self._running = True
        self._peak_mb = 0.0
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return self._peak_mb
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        return self._peak_mb


class Config:

    def __init__(self):
        # 数据集与特征处理
        self.DATASET_NAME = "cora"
        # twitter  CA-HepTh  CAGrQc  Powergrid  Hamster  cora   LastFM   Chicago

        self.PROPAGATION_THRESHOLD_MULTIPLE = 1.25
        self.LOG_TRANSFORM = True
        self.NORMALIZE_FEATURES = True

        # 伪标签与少量真实标签
        self.FEW_LABELS_RATIO = 0.02  # 微调节点比例（当 LABELS_NUM=0 时使用）
        self.LABELS_NUM = 200  # 微调节点具体数量：0=按比例确定，>0=使用此数量
        self.FEW_LABELS_STRATEGY = "stratified"  # stratified  random 建议使用分层采样以增强Top-K覆盖
        
        # 伪标签生成策略
        self.PSEUDO_LABEL_METHOD = "degree_kshell_hindex_pr"  # 选项: "degree" | "degree_cc_hindex" | "degree_kshell_hindex_pr"
        # - "degree": 仅使用度数作为伪标签
        # - "degree_cc_hindex": 使用 Degree + ClusteringCoeff + H-Index 的平均，'''肯德尔较好'''
        # - "degree_kshell_hindex_pr": 使用 Degree + K-Shell + H-Index + PageRank 的平均, '''top-k较好'''

        # Student 设置
        self.STUDENT_ENABLE = True
        self.STUDENT_TYPE = "sage"  # 仅保留 GraphSAGE 学生
        self.STUDENT_HIDDEN = 256
        self.STUDENT_DROPOUT = 0.4
        self.STUDENT_SGC_K = 2  # 仅用于 SGC
        self.STUDENT_SAGE_LAYERS = 2  # GraphSAGE 层数
        self.STUDENT_SAGE_AGGREGATOR = 'max'  # mean | sum | max
        # 蒸馏损失权重与采样
        self.KD_T = 1.0
        self.KD_NUM_PAIRS = 30000
        self.KD_TOPK = 50
        self.KD_ALPHA_GLOBAL = 0.8
        self.KD_ALPHA_TOPK = 0.2
        self.KD_LAMBDA_LIST = 0
        self.KD_LAMBDA_GT = 0.1
        self.KD_GT_USE_RANKLOSS = True
        self.KD_USE_MARGIN_WEIGHT = True
        self.KD_MARGIN_CLIP = 5.0

        # Student checkpoint 选择
        self.STUDENT_SAVE_BY = "topk"  # kendall | topk | constraint
        self.STUDENT_KENDALL_FLOOR_RATIO = 0.98

        # 教师模型结构
        self.TEACHER_GNN_TYPE = "sage"   # gcn | sage | gat | sgc
        self.TEACHER_GNN_LAYERS = 4
        self.TEACHER_HIDDEN_DIM = 128
        self.TEACHER_DROPOUT = 0.4
        # 额外的编码器超参数
        self.TEACHER_GAT_HEADS = 4
        self.TEACHER_SAGE_AGGREGATOR = 'mean'  # mean|sum|max
        self.TEACHER_SGC_K = 4

        # 训练超参
        self.SEED = 3407
        self.LR = 0.001
        self.WEIGHT_DECAY = 5e-4
        self.FINETUNE_WEIGHT_DECAY = 1e-3  # 微调阶段增强正则化，防止过拟合少量标签
        self.TEACHER_TOTAL_EPOCHS = 200
        self.PRETRAIN_EPOCHS = 200  # 分离：预训练轮数
        self.FINETUNE_EPOCHS = 500  # 分离：微调轮数
        self.ENABLE_PRETRAINING = True
        # 损失调度控制：'fixed' 使用 TEACHER_LOSS_CONFIG，'anneal' 使用进度退火
        self.LOSS_SCHEDULE_MODE = 'anneal'  # 'fixed' | 'anneal'
        # pairwise 强化结束进度（0~1），控制从 listMLE 向 pairwise 过渡的时机
        self.PAIRWISE_ANNEAL_END = 0.5

        # 分层采样（平衡Top-K覆盖与全局代表性）
        self.STRATIFIED_SAMPLING = {
            'tier1_top_k': 30,
            'tier1_ratio': 0.75,
            'tier2_top_k': 80,
            'tier2_ratio': 0.55,
            'tier3_top_k': 300,
            'tier3_ratio': 0.30,
            'tier4_ratio': 0.03,
        }

        # 教师损失配置（listMLE + pairwise，支持Top-K与分层采样）
        self.TEACHER_LOSS_CONFIG = {
            'listmle_weight': 0 , # 0.7
            'pairwise_weight': 1 , # 0.3
            'mse_weight': 0.0,
            'sample_size': 50000,
            'hard_negatives': True,
            'top_k': 100,
            'use_stratified': False, # 是否走分层 pairwise
        }

        # 可视化与阶段管理（仅用于评估与日志，不影响性能）
        self.USE_VISUALIZATION = False
        self.MULTI_STAGE_TRAINING = {
            'enable': True,
            'stage1': {
                'name': '全局排序建立',
                'epochs_ratio': 0.30,
            },
            'stage2': {
                'name': '平衡优化',
                'epochs_ratio': 0.40,
            },
            'stage3': {
                'name': 'Top-K精炼',
                'epochs_ratio': 0.30,
            },
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 强制 PyTorch 使用确定性算法，解决 Scatter 操作和其他 GPU 操作的随机性
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalize_labels(y, min_val=None, max_val=None):
    """归一化标签到[0, 1]范围"""
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
        normalized_np, min_val, max_val = normalize_labels(y_np, min_val, max_val)
        return torch.tensor(normalized_np, dtype=y.dtype, device=y.device), min_val, max_val

    y_min = min_val if min_val is not None else np.min(y)
    y_max = max_val if max_val is not None else np.max(y)
    if y_max - y_min < 1e-8:
        return y, y_min, y_max
    normalized_y = (y - y_min) / (y_max - y_min)
    return normalized_y, y_min, y_max


def denormalize_labels(y, min_val, max_val):
    """将归一化标签反归一化回原始范围"""
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
        denormalized_np = denormalize_labels(y_np, min_val, max_val)
        return torch.tensor(denormalized_np, dtype=y.dtype, device=y.device)
    return y * (max_val - min_val) + min_val


class GNN_encoder(nn.Module):
    """通用GNN编码器（PyG 版本），支持 GCN/GAT/GraphSAGE/SGC"""
    def __init__(self, in_feats, hid_feats, out_feats, layer_nums=2, dropout=0.2,
                 gnn_type="gcn", gat_heads=4, sage_aggregator='mean', sgc_k=2):
        super(GNN_encoder, self).__init__()
        self.layer_nums = layer_nums
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        if gnn_type == "gcn":
            if layer_nums == 1:
                self.conv_layers.append(PYG_GCNConv(in_feats, out_feats, normalize=True))
            else:
                self.conv_layers.append(PYG_GCNConv(in_feats, hid_feats, normalize=True))
                self.bn_layers.append(nn.BatchNorm1d(hid_feats))
                for _ in range(1, layer_nums - 1):
                    self.conv_layers.append(PYG_GCNConv(hid_feats, hid_feats, normalize=True))
                    self.bn_layers.append(nn.BatchNorm1d(hid_feats))
                self.conv_layers.append(PYG_GCNConv(hid_feats, out_feats, normalize=True))
        elif gnn_type == "gat":
            # 使用 concat=False 保持维度为 hidden_dim
            if layer_nums == 1:
                self.conv_layers.append(GATConv(in_feats, out_feats, heads=gat_heads, concat=False))
            else:
                self.conv_layers.append(GATConv(in_feats, hid_feats, heads=gat_heads, concat=False))
                self.bn_layers.append(nn.BatchNorm1d(hid_feats))
                for _ in range(1, layer_nums - 1):
                    self.conv_layers.append(GATConv(hid_feats, hid_feats, heads=gat_heads, concat=False))
                    self.bn_layers.append(nn.BatchNorm1d(hid_feats))
                self.conv_layers.append(GATConv(hid_feats, out_feats, heads=gat_heads, concat=False))
        elif gnn_type == "sage":
            if layer_nums == 1:
                self.conv_layers.append(PYG_SAGEConv(in_feats, out_feats, aggr=sage_aggregator))
            else:
                self.conv_layers.append(PYG_SAGEConv(in_feats, hid_feats, aggr=sage_aggregator))
                self.bn_layers.append(nn.BatchNorm1d(hid_feats))
                for _ in range(1, layer_nums - 1):
                    self.conv_layers.append(PYG_SAGEConv(hid_feats, hid_feats, aggr=sage_aggregator))
                    self.bn_layers.append(nn.BatchNorm1d(hid_feats))
                self.conv_layers.append(PYG_SAGEConv(hid_feats, out_feats, aggr=sage_aggregator))
        elif gnn_type == "sgc":
            # SGC 通常一层足够，采用折叠卷积 K 次
            self.layer_nums = 1
            self.conv_layers.append(SGConv(in_feats, out_feats, K=sgc_k))
        else:
            raise ValueError(f"不支持的 gnn_type: {gnn_type}")

        if self.layer_nums > 1 and gnn_type != "sgc":
            self.residual_proj.append(nn.Linear(in_feats, hid_feats) if in_feats != hid_feats else None)
            for _ in range(1, self.layer_nums - 1):
                self.residual_proj.append(None)
            self.residual_proj.append(nn.Linear(hid_feats, out_feats) if hid_feats != out_feats else None)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, conv in enumerate(self.conv_layers):
            residual = h
            # 按类型传递 edge_weight
            if isinstance(conv, (PYG_GCNConv, SGConv)):
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)

            if i < self.layer_nums - 1 and self.gnn_type != "sgc":
                h = self.bn_layers[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            if i < len(self.residual_proj) and self.gnn_type != "sgc":
                proj = self.residual_proj[i]
                if proj is not None:
                    residual = proj(residual)
                h = h + residual
        return h


class Predictor(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, dropout=0.2):
        super(Predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class StudentMLP(nn.Module):
    def __init__(self, in_feats, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)  # [N, 1]

    def predict(self, x, graph=None):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class StudentSGCMLP(nn.Module):
    def __init__(self, in_feats, hidden_dim=128, dropout=0.2, K=4):
        super().__init__()
        self.sgc = SGConv(in_feats, hidden_dim, K=K)
        self.pred = Predictor(hidden_dim, dropout=dropout)

    def forward(self, x, graph):
        z = self.sgc(x, graph['edge_index'], graph.get('edge_weight'))
        z = F.relu(z)
        z = F.dropout(z, p=0.2, training=self.training)
        y = self.pred(z)
        return y, z

    def predict(self, x, graph):
        self.eval()
        with torch.no_grad():
            y, _ = self.forward(x, graph)
        return y


class StudentSAGEMLP(nn.Module):
    """GraphSAGE + MLP 学生模型"""
    def __init__(self, in_feats, hidden_dim=128, dropout=0.2, num_layers=2, aggregator='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.sage_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # 构建 GraphSAGE 层
        for i in range(num_layers):
            input_dim = in_feats if i == 0 else hidden_dim
            output_dim = hidden_dim
            self.sage_layers.append(PYG_SAGEConv(input_dim, output_dim, aggr=aggregator))
            if i < num_layers - 1:
                self.bn_layers.append(nn.BatchNorm1d(output_dim))
        
        self.pred = Predictor(hidden_dim, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, graph):
        z = x
        for i, sage_layer in enumerate(self.sage_layers):
            z = sage_layer(z, graph['edge_index'])
            if i < self.num_layers - 1:
                z = self.bn_layers[i](z)
                z = F.relu(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        y = self.pred(z)
        return y, z

    def predict(self, x, graph):
        self.eval()
        with torch.no_grad():
            y, _ = self.forward(x, graph)
        return y


class RankingLosses:
    """排序损失函数集合（listMLE + 加权pairwise + 分层pairwise）"""
    @staticmethod
    def listmle_loss(pred, target, mask=None, top_k=50):
        if mask is not None:
            pred = pred[mask.bool()]
            target = target[mask.bool()]
        if target.dim() == 0:
            return torch.tensor(0.0, device=pred.device)
        elif target.dim() > 1:
            target = target.view(-1)
            pred = pred.view(-1)
        if len(target) == 0:
            return torch.tensor(0.0, device=pred.device)
        k_value = max(1, min(top_k, len(pred)))
        _, pred_top_indices = torch.topk(pred, k=k_value)
        pred_topk = pred[pred_top_indices]
        target_topk = target[pred_top_indices]
        loss = 0.0
        N = len(pred_topk)
        if N == 0:
            return torch.tensor(0.0, device=pred.device)
        _, target_rank_indices = torch.sort(target_topk, descending=True)
        sorted_pred = pred_topk[target_rank_indices]
        for i in range(N):
            remaining_scores = sorted_pred[i:]
            log_prob = sorted_pred[i] - torch.logsumexp(remaining_scores, dim=0)
            loss -= log_prob
        return loss / N

    @staticmethod
    def weighted_pairwise_loss(pred, target, mask=None, k=20, sample_size=None, hard_negatives=False):
        if pred.dim() == 2:
            pred = pred.squeeze(1)
        if target.dim() == 2:
            target = target.squeeze(1)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.squeeze(1)
            mask = mask.bool()
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) < 2:
                return 0.0 * pred.sum()
            pred = pred[valid_indices]
            target = target[valid_indices]

        N = pred.size(0)
        if N < 2:
            return 0.0 * pred.sum()
        total_pairs = N * (N - 1)
        if sample_size is not None and sample_size < total_pairs:
            if hard_negatives:
                k = min(50, N // 10)
                _, top_k_indices = torch.topk(pred, k)
                num_topk_pairs = sample_size // 2
                candidate_multiplier = 2
                i_topk = top_k_indices[torch.randint(0, k, (num_topk_pairs * candidate_multiplier,), device=pred.device)]
                j_topk = torch.randint(0, N, (num_topk_pairs * candidate_multiplier,), device=pred.device)
                mask_topk = i_topk != j_topk
                i_topk = i_topk[mask_topk][:num_topk_pairs]
                j_topk = j_topk[mask_topk][:num_topk_pairs]
                num_random_pairs = sample_size - num_topk_pairs
                i_random = torch.randint(0, N, (num_random_pairs * candidate_multiplier,), device=pred.device)
                j_random = torch.randint(0, N, (num_random_pairs * candidate_multiplier,), device=pred.device)
                mask_random = i_random != j_random
                i_random = i_random[mask_random][:num_random_pairs]
                j_random = j_random[mask_random][:num_random_pairs]
                idx_i = torch.cat([i_topk, i_random])
                idx_j = torch.cat([j_topk, j_random])
            else:
                candidate_multiplier = 2
                i_candidate = torch.randint(0, N, (sample_size * candidate_multiplier,), device=pred.device)
                j_candidate = torch.randint(0, N, (sample_size * candidate_multiplier,), device=pred.device)
                mask_ij = i_candidate != j_candidate
                idx_i = i_candidate[mask_ij][:sample_size]
                idx_j = j_candidate[mask_ij][:sample_size]
        else:
            idx_i = torch.arange(N).repeat(N)
            idx_j = torch.arange(N).repeat_interleave(N)
            mask_pairs = idx_i != idx_j
            idx_i = idx_i[mask_pairs]
            idx_j = idx_j[mask_pairs]

        pred_i = pred[idx_i]
        pred_j = pred[idx_j]
        target_i = target[idx_i]
        target_j = target[idx_j]
        target_labels = (target_i > target_j).float()
        pred_diff = pred_i - pred_j

        pred_full = pred.clone()
        if mask is not None:
            full_mask = torch.ones_like(pred_full, dtype=torch.bool)
            pred_full = pred_full[full_mask]
        if len(pred_full) > k:
            # 使用 topk+min 代替 kthvalue，保持等价且确定性
            topk_vals, _ = torch.topk(pred_full, k)
            threshold = topk_vals.min()
            is_top_k = (pred_i >= threshold) | (pred_j >= threshold)
        else:
            is_top_k = torch.ones_like(target_i, dtype=torch.bool)
        top_k_weight = torch.where(is_top_k, 2.0, 0.5)
        relevance_diff = torch.abs(target_i - target_j)
        weight = relevance_diff * top_k_weight
        loss = F.binary_cross_entropy_with_logits(pred_diff, target_labels, reduction='none')
        loss = (loss * weight).mean()
        return loss

    @staticmethod
    def stratified_pairwise_loss(pred, target, mask=None, sample_size=50000):
        if pred.dim() == 2:
            pred = pred.squeeze(1)
        if target.dim() == 2:
            target = target.squeeze(1)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.squeeze(1)
            mask = mask.bool()
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) < 2:
                return 0.0 * pred.sum()
            pred = pred[valid_indices]
            target = target[valid_indices]
        N = pred.size(0)
        if N < 2:
            return 0.0 * pred.sum()
        sorted_indices = torch.argsort(pred, descending=True)
        tier1_size = min(10, N)
        tier2_size = min(20, N - tier1_size)
        tier3_size = min(70, N - tier1_size - tier2_size)
        tier1_indices = sorted_indices[:tier1_size]
        tier2_indices = sorted_indices[tier1_size:tier1_size + tier2_size]
        tier3_indices = sorted_indices[tier1_size + tier2_size:tier1_size + tier2_size + tier3_size]
        tier_weights = {'tier1': 10.0, 'tier2': 5.0, 'tier3': 2.0, 'others': 1.0}
        node_tiers = torch.zeros(N, dtype=torch.int, device=pred.device)
        node_weights = torch.ones(N, dtype=torch.float, device=pred.device)
        node_tiers[tier1_indices] = 1
        node_weights[tier1_indices] = tier_weights['tier1']
        node_tiers[tier2_indices] = 2
        node_weights[tier2_indices] = tier_weights['tier2']
        node_tiers[tier3_indices] = 3
        node_weights[tier3_indices] = tier_weights['tier3']
        node_weights_sum = node_weights.sum()
        if node_weights_sum == 0:
            return 0.0 * pred.sum()
        node_prob = node_weights / node_weights_sum
        candidate_multiplier = 2
        num_candidates = sample_size * candidate_multiplier
        idx_i_candidates = torch.multinomial(node_prob, num_samples=num_candidates, replacement=True)
        idx_j_candidates = torch.multinomial(node_prob, num_samples=num_candidates, replacement=True)
        valid_mask = idx_i_candidates != idx_j_candidates
        idx_i = idx_i_candidates[valid_mask][:sample_size]
        idx_j = idx_j_candidates[valid_mask][:sample_size]
        if len(idx_i) < sample_size:
            remaining = sample_size - len(idx_i)
            random_i = torch.randint(0, N, (remaining,), device=pred.device)
            random_j = torch.randint(0, N, (remaining,), device=pred.device)
            valid_mask2 = random_i != random_j
            invalid_indices = torch.where(~valid_mask2)[0]
            while len(invalid_indices) > 0:
                random_j[invalid_indices] = torch.randint(0, N, (len(invalid_indices),), device=pred.device)
                valid_mask2 = random_i != random_j
                invalid_indices = torch.where(~valid_mask2)[0]
            idx_i = torch.cat([idx_i, random_i])
            idx_j = torch.cat([idx_j, random_j])
        pred_i = pred[idx_i]
        pred_j = pred[idx_j]
        target_i = target[idx_i]
        target_j = target[idx_j]
        target_labels = (target_i > target_j).float()
        pred_diff = pred_i - pred_j
        tier_i = node_tiers[idx_i]
        tier_j = node_tiers[idx_j]
        max_tier = torch.max(tier_i, tier_j)
        pair_tier_weights = torch.ones_like(max_tier, dtype=torch.float, device=pred.device)
        pair_tier_weights[max_tier == 1] = tier_weights['tier1']
        pair_tier_weights[max_tier == 2] = tier_weights['tier2']
        pair_tier_weights[max_tier == 3] = tier_weights['tier3']
        loss = F.binary_cross_entropy_with_logits(pred_diff, target_labels, reduction='none')
        loss = (loss * pair_tier_weights).mean()
        return loss

    @staticmethod
    def mixed_ranking_loss_v2(pred, target, mask=None,
                               listmle_weight=0.7,
                               pairwise_weight=0.3,
                               top_k=50,
                               use_stratified=False,
                               sample_size=10000):
        listmle = RankingLosses.listmle_loss(pred, target, mask, top_k)
        if use_stratified:
            pairwise = RankingLosses.stratified_pairwise_loss(pred, target, mask, sample_size=sample_size)
        else:
            pairwise = RankingLosses.weighted_pairwise_loss(
                pred, target, mask, k=20, sample_size=sample_size, hard_negatives=True
            )
        return listmle_weight * listmle + pairwise_weight * pairwise


class PretrainBestSelector:
    """
    仅基于伪标签与结构一致性选择最优预训练 checkpoint
    ✅ 不涉及任何真实标签（无标签泄露）
    """
    def __init__(self, pseudo_labels, top_k=50, min_epoch=50):
        self.pseudo_labels = pseudo_labels.cpu().numpy().squeeze()
        self.top_k = top_k
        self.min_epoch = min_epoch

        self.best_score = -np.inf
        self.best_state = None
        self.best_epoch = -1
        self.prev_topk = None

    @torch.no_grad()
    def evaluate(self, scores, epoch):
        """
        评估当前 checkpoint（基于伪标签与结构稳定性）
        scores: Tensor [N]
        """
        scores_np = scores.cpu().numpy().squeeze()

        # A. Pseudo-Kendall：模型排序 vs 伪标签排序的一致性
        try:
            tau, _ = kendalltau(scores_np, self.pseudo_labels)
            tau = 0.0 if np.isnan(tau) else tau
        except:
            tau = 0.0

        # B. Top-K 稳定性：当前 Top-K 与前一轮的 Jaccard
        cur_topk = set(np.argsort(scores_np)[-self.top_k:])
        if self.prev_topk is None:
            stability = 0.0
        else:
            intersection = len(cur_topk & self.prev_topk)
            stability = intersection / self.top_k
        self.prev_topk = cur_topk

        # C. 结构组合分数（可调权重）
        combined = 0.7 * tau + 0.3 * stability

        # D. 更新最优 checkpoint
        if epoch >= self.min_epoch and combined > self.best_score:
            self.best_score = combined
            self.best_epoch = epoch

        return {
            "pseudo_kendall": tau,
            "topk_stability": stability,
            "combined": combined
        }

    def update_best_state(self, model_state):
        """保存当前最优权重"""
        if self.best_epoch >= self.min_epoch:
            self.best_state = deepcopy(model_state)


class RankingObjective:
    """统一的排序目标：提供损失与权重调度，确保配置可控"""
    def __init__(self, config: Config):
        self.config = config
        self.losses = RankingLosses()

    def get_loss_schedule(self, epoch: int, total_epochs: int, phase: str, topk_acc: float = None):
        mode = getattr(self.config, 'LOSS_SCHEDULE_MODE', 'anneal')
        base = self.config.TEACHER_LOSS_CONFIG.copy()
        # fixed 模式：完全遵循配置
        if mode == 'fixed':
            return base

        # anneal 模式：按进度退火，使用配置作为基准
        progress = epoch / (max(total_epochs - 1, 1))
        # 三段式：建立 → 过渡 → 强化 Top-K
        if progress < 0.2:
            listmle_weight = base.get('listmle_weight', 0.7)
            pairwise_weight = base.get('pairwise_weight', 0.3)
            top_k = base.get('top_k', 50)
            use_stratified = False
        elif progress < max(self.config.PAIRWISE_ANNEAL_END, 0.2):
            phase_progress = (progress - 0.2) / max(self.config.PAIRWISE_ANNEAL_END - 0.2, 1e-6)
            # 从 base 向 pairwise 强化过渡（上限 0.8）
            pairwise_weight = min(0.8, base.get('pairwise_weight', 0.3) + phase_progress * 0.5)
            listmle_weight = max(0.2, base.get('listmle_weight', 0.7) - phase_progress * 0.5)
            top_k = int(base.get('top_k', 50) - phase_progress * 30)
            use_stratified = True
        else:
            listmle_weight = 0.2
            pairwise_weight = 0.8
            top_k = 20
            use_stratified = True

        schedule = base.copy()
        schedule['listmle_weight'] = float(listmle_weight)
        schedule['pairwise_weight'] = float(pairwise_weight)
        schedule['top_k'] = int(top_k)
        schedule['use_stratified'] = bool(use_stratified)
        return schedule

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, schedule: dict):
        return self.losses.mixed_ranking_loss_v2(
            pred, target, mask,
            listmle_weight=schedule.get('listmle_weight', 0.7),
            pairwise_weight=schedule.get('pairwise_weight', 0.3),
            top_k=schedule.get('top_k', 50),
            use_stratified=schedule.get('use_stratified', False),
            sample_size=schedule.get('sample_size', 50000),
        )

class RankingGNNModel(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_layers, dropout=0.1,
                 gnn_type="gcn", gat_heads=4, sage_aggregator='mean', sgc_k=2):
        super(RankingGNNModel, self).__init__()
        self.encoder = GNN_encoder(
            in_feats=in_feats,
            hid_feats=hidden_dim,
            out_feats=hidden_dim,
            layer_nums=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            gat_heads=gat_heads,
            sage_aggregator=sage_aggregator,
            sgc_k=sgc_k,
        )
        self.predictor = Predictor(hidden_dim, dropout=dropout)
        self.ranking_losses = RankingLosses()

    def forward(self, x, graph, y=None, label_mask=None, phase="train", loss_config=None):
        z = self.encoder(x, graph['edge_index'], graph.get('edge_weight'))
        pred = self.predictor(z)
        ranking_loss = 0.0
        if y is not None:
            default_config = {
                'listmle_weight': 0.7,
                'pairwise_weight': 0.3,
                'mse_weight': 0.0,
                'top_k': 50,
                'use_stratified': False,
                'sample_size': 50000
            }
            if loss_config is None:
                loss_config = default_config
            else:
                default_config.update(loss_config)
                loss_config = default_config
            if phase == "pretrain":
                ranking_loss = self.ranking_losses.mixed_ranking_loss_v2(
                    pred, y, None,
                    listmle_weight=loss_config['listmle_weight'],
                    pairwise_weight=loss_config['pairwise_weight'],
                    top_k=loss_config.get('top_k', 50),
                    use_stratified=loss_config.get('use_stratified', False),
                    sample_size=loss_config.get('sample_size', 50000)
                )
            else:
                ranking_loss = self.ranking_losses.mixed_ranking_loss_v2(
                    pred, y, label_mask.bool() if label_mask is not None else None,
                    listmle_weight=loss_config['listmle_weight'],
                    pairwise_weight=loss_config['pairwise_weight'],
                    top_k=loss_config.get('top_k', 50),
                    use_stratified=loss_config.get('use_stratified', False),
                    sample_size=loss_config.get('sample_size', 50000)
                )
        total_loss = ranking_loss
        return total_loss, pred

    def predict(self, x, graph):
        self.eval()
        with torch.no_grad():
            z = self.encoder(x, graph['edge_index'], graph.get('edge_weight'))
            pred = self.predictor(z)
        return pred

    def get_embeddings(self, x, graph):
        self.eval()
        with torch.no_grad():
            z = self.encoder(x, graph['edge_index'], graph.get('edge_weight'))
        return z


def calculate_topk_accuracy(pred, y_real, k=10):
    real_topk_indices = np.argsort(y_real)[-k:]
    pred_topk_indices = np.argsort(pred)[-k:]
    overlap = len(set(real_topk_indices) & set(pred_topk_indices)) / k
    return overlap


def calculate_jaccard(pred, y_real, k=10):
    real_topk_indices = np.argsort(y_real)[-k:]
    pred_topk_indices = np.argsort(pred)[-k:]
    intersection = len(set(real_topk_indices) & set(pred_topk_indices))
    union = len(set(real_topk_indices) | set(pred_topk_indices))
    if union == 0:
        return 0.0
    return intersection / union


def calculate_imprecision(y_pred, y_true, k):
    """Imprecision Index (epsilon), 越小越好。"""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    top_k_pred_indices = np.argsort(y_pred)[-k:][::-1]
    m_alg = float(np.mean(y_true[top_k_pred_indices]))
    top_k_true_values = np.sort(y_true)[-k:]
    m_opt = float(np.mean(top_k_true_values))
    if abs(m_opt) < 1e-12:
        return 0.0
    epsilon = 1.0 - (m_alg / m_opt)
    return float(epsilon)


def calculate_ndcg(pred, y_real, k=20):
    pred = np.squeeze(pred)
    y_real = np.squeeze(y_real)
    n = len(y_real)
    k = min(k, n)
    if k == 0:
        return 0.0
    return ndcg_score(y_real.reshape(1, -1), pred.reshape(1, -1), k=k)


def calculate_monotonicity_index(pred, y_real=None):
    from collections import Counter
    pred = np.squeeze(pred)
    N = len(pred)
    if N <= 1:
        return 1.0
    pred_clean = []
    for val in pred:
        if not np.isnan(val):
            pred_clean.append(val)
        else:
            pred_clean.append(f"NaN_{id(val)}")
    pred_counts = Counter(pred_clean)
    numerator = sum(N_a * (N_a - 1) for N_a in pred_counts.values())
    denominator = N * (N - 1)
    if denominator == 0:
        return 1.0
    mi = (1 - numerator / denominator) ** 2
    return mi


def evaluate_model(model, graph, features, y_real, min_val=None, max_val=None, eval_topk=[10, 20, 30], eval_ndcg=[10, 20, 30], use_double=False):
    """评估模型性能；可选使用 float64 精度（默认关闭以提升速度）"""
    model.eval()
    if use_double:
        features = features.double()
        model = model.double()
    
    with torch.no_grad():
        pred = model.predict(features, graph)
    
    # 若使用了 double，则转回 float32，避免后续计算额外开销
    if use_double and pred.dtype == torch.float64:
        pred = pred.float()
    
    pred_np = pred.cpu().numpy().squeeze()
    y_real_np = y_real.cpu().numpy().squeeze()
    if min_val is not None and max_val is not None:
        pred_np_denorm = denormalize_labels(pred_np, min_val, max_val)
        y_real_np_denorm = denormalize_labels(y_real_np, min_val, max_val)
    else:
        pred_np_denorm = pred_np
        y_real_np_denorm = y_real_np
    try:
        kendall, _ = kendalltau(pred_np_denorm, y_real_np_denorm)
    except Exception:
        kendall = -1.0
    mse = mean_squared_error(y_real_np_denorm, pred_np_denorm)
    try:
        spearman, _ = spearmanr(pred_np_denorm, y_real_np_denorm)
    except Exception:
        spearman = -1.0
    topk_acc = {f'top{k}_acc': calculate_topk_accuracy(pred_np_denorm, y_real_np_denorm, k=k) for k in eval_topk}
    topk_jaccard = {f'top{k}_jaccard': calculate_jaccard(pred_np_denorm, y_real_np_denorm, k=k) for k in eval_topk}
    ndcg_at_k = {f'ndcg_at_{k}': calculate_ndcg(pred_np_denorm, y_real_np_denorm, k=k) for k in eval_ndcg}
    imprecision_at_k = {f'imprecision_at_{k}': calculate_imprecision(pred_np_denorm, y_real_np_denorm, k=k) for k in eval_topk}
    mi = calculate_monotonicity_index(pred_np_denorm, y_real_np_denorm)
    results = {"kendall": kendall, "mse": mse, "spearman": spearman, "mi": mi}
    results.update(topk_acc)
    results.update(topk_jaccard)
    results.update(ndcg_at_k)
    results.update(imprecision_at_k)
    
    # 恢复原始精度
    if use_double:
        model.float()
    
    return results


def calculate_inference_time(model, graph, features, device, repetitions=100):
    """
    科学测量模型的推理时间（符合Knowledge Distillation标准）
    
    Args:
        model: PyTorch模型
        graph: 图结构（包含edge_index等）
        features: 节点特征
        device: 'cuda' or 'cpu'
        repetitions: 重复测量次数，默认100
    
    Returns:
        avg_time_ms: 平均推理时间（毫秒）
        avg_time_s: 平均推理时间（秒）
    """
    model.eval()
    
    # 1. 预热（Warm-up）：GPU第一次运行需要初始化CUDA上下文，必须排除
    with torch.no_grad():
        for _ in range(10):
            _ = model.predict(features, graph)
            if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
                torch.cuda.synchronize()
    
    # 2. 正式测量：多次运行取平均，消除系统抖动
    timings = []
    with torch.no_grad():
        for _ in range(repetitions):
            # 确保之前的GPU操作全部完成
            if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
                torch.cuda.synchronize()
            
            # 计时开始
            start = time.perf_counter()
            
            # 模型推理（纯前向传播）
            _ = model.predict(features, graph)
            
            # 等待GPU计算完成（PyTorch GPU操作是异步的）
            if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
                torch.cuda.synchronize()
            
            # 计时结束
            end = time.perf_counter()
            
            timings.append(end - start)
    
    avg_time_s = np.mean(timings)
    std_time_s = np.std(timings)
    avg_time_ms = avg_time_s * 1000
    
    return avg_time_ms, avg_time_s


def plot_kendall_trend(epochs, kendalls, save_path, best_point=None):
    plt.figure()
    plt.plot(epochs, kendalls, marker='o')
    if best_point is not None:
        plt.scatter(best_point['epoch'], best_point['value'], color='C0', marker='*', s=120, zorder=5)
        plt.annotate(f"best {best_point['value']:.3f}", (best_point['epoch'], best_point['value']),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='C0')
    plt.xlabel('Epoch')
    plt.ylabel('Kendall Tau')
    plt.title('Kendall Trend')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_topk_trend(epochs, top20, top50, top100, save_path):
    """Plot Top-20/50/100 Jaccard trends on one figure with best points annotated."""
    plt.figure()
    plt.plot(epochs, top20, marker='o', label='Top-20 Jaccard')
    plt.plot(epochs, top50, marker='s', label='Top-50 Jaccard')
    plt.plot(epochs, top100, marker='^', label='Top-100 Jaccard')

    # annotate best points
    def annotate_best(xs, ys, color, label):
        if len(xs) == 0:
            return
        best_idx = int(np.argmax(ys))
        plt.scatter(xs[best_idx], ys[best_idx], color=color, marker='*', s=120, zorder=5)
        plt.annotate(f"best {ys[best_idx]:.3f}", (xs[best_idx], ys[best_idx]),
                     textcoords="offset points", xytext=(0, 10), ha='center', color=color)

    annotate_best(epochs, top20, 'C0', 'Top-20')
    annotate_best(epochs, top50, 'C1', 'Top-50')
    annotate_best(epochs, top100, 'C2', 'Top-100')

    plt.xlabel('Epoch')
    plt.ylabel('Jaccard')
    plt.title('Top-K Jaccard Trend')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_kendall_trend_compare(with_epochs, with_kendall, without_epochs, without_kendall, pretrain_epochs, save_path,
                               best_with=None, best_without=None):
    """Single figure comparing with-pretrain vs no-pretrain, x-axis aligned by total epochs.

    best_with: {'epoch': int, 'value': float} for Kendall best point (already aligned epoch)
    best_without: {'epoch': int, 'value': float} for Kendall best point (unshifted, will be aligned inside)
    """
    plt.figure()
    plt.plot(with_epochs, with_kendall, marker='o', label='With Pretrain')
    aligned_without_epochs = [e + pretrain_epochs for e in without_epochs]
    plt.plot(aligned_without_epochs, without_kendall, marker='s', label='No Pretrain (Shifted)')
    plt.axvline(pretrain_epochs, color='gray', linestyle='--', alpha=0.5, label=f'Finetune start @ {pretrain_epochs}')

    # 标记历史最佳点（Kendall）
    if best_with is not None:
        plt.scatter(best_with['epoch'], best_with['value'], color='C0', marker='*', s=120, zorder=5)
        plt.annotate(f"best {best_with['value']:.3f}", (best_with['epoch'], best_with['value']),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='C0')
    if best_without is not None:
        aligned_epoch = best_without['epoch'] + pretrain_epochs
        plt.scatter(aligned_epoch, best_without['value'], color='C1', marker='*', s=120, zorder=5)
        plt.annotate(f"best {best_without['value']:.3f}", (aligned_epoch, best_without['value']),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='C1')

    plt.xlabel('Epoch')
    plt.ylabel('Kendall Tau')
    plt.title('Kendall Trend: Pretrain vs No Pretrain')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def sample_pairs_global(N, num_pairs, device):
    i = torch.randint(0, N, (num_pairs,), device=device)
    j = torch.randint(0, N, (num_pairs,), device=device)
    mask = i != j
    return i[mask], j[mask]


def plot_student_kendall_trends(student_results):
    """
    绘制多个学生模型的 Kendall 系数变化趋势，标注最高点。
    使用 plt.show() 显示而不保存。
    """
    plt.figure(figsize=(10, 6))
    
    colors = {'mlp': 'C0', 'sgc': 'C1', 'sage': 'C2'}
    markers = {'mlp': 'o', 'sgc': 's', 'sage': '^'}
    
    for stype in ['mlp', 'sgc', 'sage']:
        if stype not in student_results:
            continue
        epochs = student_results[stype].get('epoch_history', [])
        kendall = student_results[stype].get('kendall_history', [])
        
        if len(epochs) == 0:
            continue
            
        plt.plot(epochs, kendall, marker=markers[stype], label=f'Student-{stype}', 
                color=colors[stype], linewidth=2, markersize=6)
        
        # 标注最高点
        best_idx = int(np.argmax(kendall))
        plt.scatter(epochs[best_idx], kendall[best_idx], color=colors[stype], 
                   marker='*', s=200, zorder=5, edgecolors='black', linewidths=0.5)
        plt.annotate(f"best {kendall[best_idx]:.4f}", 
                    (epochs[best_idx], kendall[best_idx]),
                    textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=9, color=colors[stype], weight='bold')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Kendall Tau', fontsize=12)
    plt.title('学生模型 Kendall 系数训练趋势对比', fontsize=13, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()


def plot_student_topk_trends(student_results):
    """
    绘制学生模型的 Top-10, Top-50, Top-100 Jaccard 变化趋势（3个子图）。
    使用 plt.show() 显示而不保存。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {'mlp': 'C0', 'sgc': 'C1', 'sage': 'C2'}
    markers = {'mlp': 'o', 'sgc': 's', 'sage': '^'}
    
    topk_configs = [
        ('top10_history', 'Top-10 Jaccard', axes[0]),
        ('top50_history', 'Top-50 Jaccard', axes[1]),
        ('top100_history', 'Top-100 Jaccard', axes[2])
    ]
    
    for key, title, ax in topk_configs:
        for stype in ['mlp', 'sgc', 'sage']:
            if stype not in student_results:
                continue
            epochs = student_results[stype].get('epoch_history', [])
            topk = student_results[stype].get(key, [])
            
            if len(epochs) == 0:
                continue
            
            ax.plot(epochs, topk, marker=markers[stype], label=f'Student-{stype}',
                   color=colors[stype], linewidth=2, markersize=6)
            
            # 标注最高点
            best_idx = int(np.argmax(topk))
            ax.scatter(epochs[best_idx], topk[best_idx], color=colors[stype],
                      marker='*', s=200, zorder=5, edgecolors='black', linewidths=0.5)
            ax.annotate(f"{topk[best_idx]:.3f}", 
                       (epochs[best_idx], topk[best_idx]),
                       textcoords="offset points", xytext=(0, 8), ha='center',
                       fontsize=8, color=colors[stype], weight='bold')
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Jaccard', fontsize=11)
        ax.set_title(title, fontsize=11, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10, loc='best')
    
    fig.suptitle('学生模型 Top-K 指标训练趋势', fontsize=13, weight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_student_topk_trends_save(student_model_type, epoch_history, top10_history, top50_history, top100_history, 
                                   dataset_name, save_path=None):
    """
    绘制单个学生模型的 Top-10, Top-50, Top-100 Jaccard 变化趋势，并保存为文件。
    
    Args:
        student_model_type: 学生模型类型，如 'sage'
        epoch_history: 训练 epoch 列表
        top10_history: Top-10 Jaccard 历史列表
        top50_history: Top-50 Jaccard 历史列表
        top100_history: Top-100 Jaccard 历史列表
        dataset_name: 数据集名称，用于文件名
        save_path: 保存路径，如果为 None 则使用默认路径
    """
    if save_path is None:
        save_path = f"student_{student_model_type}_topk_trends_{dataset_name}.png"
    
    # 创建单个图表展示所有三条曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制三条曲线
    ax.plot(epoch_history, top10_history, marker='o', label='Top-10 Jaccard', 
            linewidth=2, markersize=6, color='C0')
    ax.plot(epoch_history, top50_history, marker='s', label='Top-50 Jaccard', 
            linewidth=2, markersize=6, color='C1')
    ax.plot(epoch_history, top100_history, marker='^', label='Top-100 Jaccard', 
            linewidth=2, markersize=6, color='C2')
    
    # 标注每条曲线的最高点
    for history, color in [(top10_history, 'C0'), (top50_history, 'C1'), (top100_history, 'C2')]:
        if len(history) > 0:
            best_idx = int(np.argmax(history))
            best_epoch = epoch_history[best_idx]
            best_val = history[best_idx]
            ax.scatter(best_epoch, best_val, color=color, marker='*', s=300, zorder=5, 
                      edgecolors='black', linewidths=1)
            ax.annotate(f"{best_val:.4f}", (best_epoch, best_val),
                       textcoords="offset points", xytext=(0, 10), ha='center',
                       fontsize=9, color=color, weight='bold')
    
    ax.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax.set_ylabel('Jaccard Coefficient', fontsize=12, weight='bold')
    ax.set_title(f'Student-{student_model_type.upper()} Top-K Jaccard Trends ({dataset_name})', 
                fontsize=13, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    fig.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # print(f"[Visualization] 学生模型 Top-K 趋势已保存至: {save_path}")
    plt.show()


def sample_pairs_topk(teacher_scores, K, num_pairs, device):
    N = teacher_scores.numel()
    k = min(K, N)
    _, top_idx = torch.topk(teacher_scores, k=k)
    i = top_idx[torch.randint(0, k, (num_pairs,), device=device)]
    j = torch.randint(0, N, (num_pairs,), device=device)
    top_mask = torch.zeros(N, dtype=torch.bool, device=device)
    top_mask[top_idx] = True
    for _ in range(3):
        bad = top_mask[j]
        if bad.any():
            j[bad] = torch.randint(0, N, (bad.sum().item(),), device=device)
    mask = i != j
    return i[mask], j[mask]


def kd_pairwise_bce(student_scores, teacher_scores, idx_i, idx_j, T=1.0, use_margin_weight=True, margin_clip=5.0):
    dt = (teacher_scores[idx_i] - teacher_scores[idx_j]) / T
    ds = (student_scores[idx_i] - student_scores[idx_j]) / T
    pt = torch.sigmoid(dt).detach()
    ps = torch.sigmoid(ds)
    bce = F.binary_cross_entropy(ps, pt, reduction='none')
    if use_margin_weight:
        w = torch.abs(dt).clamp(0, margin_clip).detach()
        w = w / (w.mean() + 1e-8)
        bce = bce * w
    return bce.mean()


def kd_listwise_kl(student_scores, teacher_scores, T=1.0, batch_size=1024, steps=5):
    N = student_scores.numel()
    device = student_scores.device
    loss_sum = 0.0
    for _ in range(steps):
        idx = torch.randint(0, N, (min(batch_size, N),), device=device)
        st = student_scores[idx] / T
        tt = teacher_scores[idx] / T
        qt = F.softmax(tt, dim=0).detach()
        qs = F.log_softmax(st, dim=0)
        loss_sum += F.kl_div(qs, qt, reduction='batchmean')
    return loss_sum / steps


class MultiStageTrainingManager:
    def __init__(self, config):
        self.config = config
        self._last_reported_stage = 0

    def get_current_stage(self, epoch, total_epochs):
        if not self.config.MULTI_STAGE_TRAINING['enable']:
            return None, None
        progress = epoch / total_epochs if total_epochs > 0 else 1.0
        stages = self.config.MULTI_STAGE_TRAINING
        if progress < stages['stage1']['epochs_ratio']:
            return 1, stages['stage1']
        elif progress < stages['stage1']['epochs_ratio'] + stages['stage2']['epochs_ratio']:
            return 2, stages['stage2']
        else:
            return 3, stages['stage3']

    def check_and_report_stage_change(self, epoch, total_epochs):
        stage_num, stage_config = self.get_current_stage(epoch, total_epochs)
        is_changed = stage_num != self._last_reported_stage
        if is_changed:
            print(f"\n{'='*60}")
            print(f"切换到阶段{stage_num}: {stage_config['name']}")
            print(f"{'='*60}")
            self._last_reported_stage = stage_num
        return stage_num, stage_config, is_changed


def calculate_combined_score_detailed(results, stage=None):
    if stage == 1:
        weights = {'global_ranking': 0.50, 'ranking_quality': 0.30, 'topk_identification': 0.20}
    elif stage == 2:
        weights = {'global_ranking': 0.30, 'ranking_quality': 0.35, 'topk_identification': 0.35}
    elif stage == 3:
        weights = {'global_ranking': 0.20, 'ranking_quality': 0.30, 'topk_identification': 0.50}
    else:
        weights = {'global_ranking': 0.30, 'ranking_quality': 0.35, 'topk_identification': 0.35}
    global_score = results.get('kendall', 0.0) * 0.6 + results.get('spearman', 0.0) * 0.4
    ranking_quality_score = (
        results.get('ndcg_at_10', 0.0) * 0.40 +
        results.get('ndcg_at_20', 0.0) * 0.30 +
        results.get('ndcg_at_50', 0.0) * 0.20 +
        results.get('ndcg_at_100', 0.0) * 0.10
    )
    topk_identification_score = (
        results.get('top10_jaccard', 0.0) * 0.50 +
        results.get('top20_jaccard', 0.0) * 0.30 +
        results.get('top30_jaccard', 0.0) * 0.15 +
        results.get('top50_jaccard', 0.0) * 0.05
    )
    total_score = (
        weights['global_ranking'] * global_score +
        weights['ranking_quality'] * ranking_quality_score +
        weights['topk_identification'] * topk_identification_score
    )
    return {
        'total_score': total_score,
        'global_score': global_score,
        'ranking_score': ranking_quality_score,
        'topk_score': topk_identification_score,
        'weights': weights
    }


def generate_few_labels(y_real, ratio=0.1, count=None, selection_strategy='random', config=None, y_pseudo=None, labels_num=0):
    """生成少量标签用于微调。
    
    Args:
        y_real: 真实标签
        ratio: 标签比例（当 labels_num=0 时使用）
        count: 标签数量（优先级：labels_num > count > ratio）
        selection_strategy: 采样策略
        config: 配置对象
        y_pseudo: 伪标签
        labels_num: 微调节点数量（0=使用比例，>0=使用此数量）
    """
    n_nodes = y_real.shape[0]
    # 优先级: labels_num > count > ratio
    if labels_num > 0:
        n_labels = int(max(1, min(labels_num, n_nodes)))
    elif count is not None:
        n_labels = int(max(1, min(count, n_nodes)))
    else:
        n_labels = max(1, int(n_nodes * ratio))
    
    # 记录标签数量来源
    if labels_num > 0:
        print(f"[Label Selection] 使用显式节点数量: {n_labels} (LABELS_NUM={labels_num})")
    elif count is not None:
        print(f"[Label Selection] 使用 count 参数: {n_labels}")
    else:
        print(f"[Label Selection] 使用比例确定: {n_labels} = {n_nodes} × {ratio:.4f}")
    
    if selection_strategy == 'stratified' and config is not None:
        return generate_few_labels_stratified(y_real, n_labels, config, y_pseudo=y_pseudo)
    elif selection_strategy == 'random':
        labeled_indices = np.random.choice(n_nodes, n_labels, replace=False)
    else:
        print(f"Warning: Invalid selection strategy '{selection_strategy}', using random.")
        labeled_indices = np.random.choice(n_nodes, n_labels, replace=False)
    label_mask = torch.zeros(n_nodes, 1).to(y_real.device)
    label_mask[np.array(labeled_indices)] = 1.0
    y_few = y_real * label_mask
    return y_few, label_mask


def generate_few_labels_stratified(y_real, n_labels, config, y_pseudo=None):
    n_nodes = y_real.shape[0]
    stratified_config = config.STRATIFIED_SAMPLING
    tier1_top_k = stratified_config['tier1_top_k']
    tier1_ratio = stratified_config['tier1_ratio']
    tier2_top_k = stratified_config['tier2_top_k']
    tier2_ratio = stratified_config['tier2_ratio']
    tier3_top_k = stratified_config['tier3_top_k']
    tier3_ratio = stratified_config['tier3_ratio']
    tier4_ratio = stratified_config['tier4_ratio']
    if y_pseudo is not None:
        y_rank_np = y_pseudo.cpu().numpy().flatten()
        print("使用伪标签进行节点排序和分层")
    else:
        raise ValueError("分层采样需要伪标签 y_pseudo")
    sorted_indices = np.argsort(y_rank_np)[::-1]
    tier1_indices = sorted_indices[:tier1_top_k]
    tier2_indices = sorted_indices[tier1_top_k:tier2_top_k]
    tier3_indices = sorted_indices[tier2_top_k:tier3_top_k]
    tier4_indices = sorted_indices[tier3_top_k:]
    if n_labels <= 3:
        n_tier1 = min(n_labels, len(tier1_indices))
        n_tier2 = max(0, min(n_labels - n_tier1, len(tier2_indices)))
        n_tier3 = max(0, min(n_labels - n_tier1 - n_tier2, len(tier3_indices)))
        n_tier4 = 0
    else:
        n_tier1 = max(1, int(len(tier1_indices) * tier1_ratio))
        n_tier1 = min(n_tier1, len(tier1_indices))
        n_tier2 = max(1, int(len(tier2_indices) * tier2_ratio))
        n_tier2 = min(n_tier2, len(tier2_indices))
        n_tier3 = max(1, int(len(tier3_indices) * tier3_ratio))
        n_tier3 = min(n_tier3, len(tier3_indices))
        remaining_labels = n_labels - n_tier1 - n_tier2 - n_tier3
        n_tier4 = max(0, min(remaining_labels, len(tier4_indices)))
        if n_tier4 == 0:
            tier4_sample = max(0, int(len(tier4_indices) * tier4_ratio))
            if tier4_sample > 0:
                available_slots = max(0, n_labels - (n_tier1 + n_tier2 + n_tier3))
                n_tier4 = min(tier4_sample, available_slots, len(tier4_indices))
    total_candidate = n_tier1 + n_tier2 + n_tier3 + n_tier4
    if total_candidate > n_labels:
        reduction_ratio = n_labels / total_candidate
        n_tier1 = max(1, int(n_tier1 * reduction_ratio)) if n_tier1 > 0 else 0
        n_tier2 = max(0, int(n_tier2 * reduction_ratio)) if n_tier2 > 0 else 0
        n_tier3 = max(0, int(n_tier3 * reduction_ratio)) if n_tier3 > 0 else 0
        n_tier4 = max(0, int(n_tier4 * reduction_ratio)) if n_tier4 > 0 else 0
        total_adjusted = n_tier1 + n_tier2 + n_tier3 + n_tier4
        if total_adjusted < n_labels:
            if n_tier1 < len(tier1_indices):
                n_tier1 += min(n_labels - total_adjusted, len(tier1_indices) - n_tier1)
        elif total_adjusted > n_labels:
            if n_tier4 > 0:
                n_tier4 -= min(total_adjusted - n_labels, n_tier4)
    labeled_tier1 = np.random.choice(tier1_indices, n_tier1, replace=False) if n_tier1 > 0 and len(tier1_indices) > 0 else np.array([])
    labeled_tier2 = np.random.choice(tier2_indices, n_tier2, replace=False) if n_tier2 > 0 and len(tier2_indices) > 0 else np.array([])
    labeled_tier3 = np.random.choice(tier3_indices, n_tier3, replace=False) if n_tier3 > 0 and len(tier3_indices) > 0 else np.array([])
    labeled_tier4 = np.array([])
    if n_tier4 > 0 and len(tier4_indices) > 0:
        labeled_tier4 = np.random.choice(tier4_indices, n_tier4, replace=False)
    labeled_indices = np.concatenate([labeled_tier1, labeled_tier2, labeled_tier3, labeled_tier4])
    actual_total = len(labeled_indices)
    if actual_total > n_labels:
        labeled_indices = np.random.choice(labeled_indices, n_labels, replace=False)
        actual_total = n_labels
    elif actual_total < n_labels and n_nodes > actual_total:
        all_indices = np.arange(n_nodes)
        used_indices = set(labeled_indices)
        unused_indices = [idx for idx in all_indices if idx not in used_indices]
        if unused_indices:
            additional_indices = np.random.choice(unused_indices, min(n_labels - actual_total, len(unused_indices)), replace=False)
            labeled_indices = np.concatenate([labeled_indices, additional_indices])
            actual_total = len(labeled_indices)
    label_mask = torch.zeros(n_nodes, 1).to(y_real.device)
    label_mask[labeled_indices] = 1.0
    y_few = y_real * label_mask
    print(f"\n{'='*50}")
    print(f"分层标注采样策略 (Stratified Sampling)")
    print(f"{'='*50}")
    print(f"总标注数量: {n_labels}")
    print(f"  Tier 1 (Top-{tier1_top_k}): {len(labeled_tier1)}个标注 ({tier1_ratio*100:.0f}%)")
    print(f"  Tier 2 (Top-{tier2_top_k}): {len(labeled_tier2)}个标注 ({tier2_ratio*100:.0f}%)")
    print(f"  Tier 3 (Top-{tier3_top_k}): {len(labeled_tier3)}个标注 ({tier3_ratio*100:.0f}%)")
    print(f"  Tier 4 (Others): {len(labeled_tier4)}个标注 ({tier4_ratio*100:.1f}%)")
    print(f"{'='*50}")
    if actual_total != n_labels:
        print(f"注意: 实际标注数量 ({actual_total}) 与目标 ({n_labels}) 不一致")
    return y_few, label_mask


def run_teacher_training(config, graph, features_t, y_real_norm, y_min, y_max, y_pseudo_norm, label_mask,
                         enable_pretraining=True, run_tag="with_pretrain"):
    print(f"\n=== 训练教师模型 ({config.TEACHER_GNN_TYPE}) - {run_tag} ===")
    device = features_t.device  # 获取设备信息
    
    teacher_model = RankingGNNModel(
        in_feats=features_t.shape[1],
        hidden_dim=config.TEACHER_HIDDEN_DIM,
        num_layers=config.TEACHER_GNN_LAYERS,
        dropout=config.TEACHER_DROPOUT,
        gnn_type=config.TEACHER_GNN_TYPE,
        gat_heads=config.TEACHER_GAT_HEADS,
        sage_aggregator=config.TEACHER_SAGE_AGGREGATOR,
        sgc_k=config.TEACHER_SGC_K,
    ).to(features_t.device)

    stage_manager = MultiStageTrainingManager(config)
    objective = RankingObjective(config)

    # === 修改步骤 2: 定义需要被 Mask 的"作弊列"索引 (Input Hardness) ===
    # ⚠️ 严重注意：索引必须对应 features_t (筛选后的特征矩阵) 的列索引
    # 原始特征选择: selected_features = [0, 1, 2, 4, 7, 8, 10, 11, 12]
    # 索引映射关系（原特征 -> 新特征）:
    #   0 (Degree)    -> 新索引 0 ✅ 必须 Mask
    #   4 (PageRank)  -> 新索引 3 ✅ 必须 Mask（之前错误地用了 4）
    #   7 (K-Shell)   -> 新索引 4 ✅ 必须 Mask
    #   8 (H-Index)   -> 新索引 5 ✅ 必须 Mask（之前错误地用了 8）
    # 【关键修正】使用正确的新索引 [0, 3, 4, 5]，而不是 [0, 4, 7, 8]
    CHEAT_INDICES = [0, 3, 4, 5]  # Degree, PageRank, KShell, HIndex (CORRECTED)

    # 使用配置中分离的预训练/微调轮数，而不是固定的 TEACHER_TOTAL_EPOCHS // 2
    if enable_pretraining:
        pretrain_epochs = config.PRETRAIN_EPOCHS
        finetune_epochs = config.FINETUNE_EPOCHS
    else:
        pretrain_epochs = 0
        finetune_epochs = config.FINETUNE_EPOCHS  # 无预训练时，全部轮数用于微调

    kendall_history = []
    epoch_history = []
    top20_history = []
    top50_history = []
    top100_history = []

    if enable_pretraining:
        optimizer_pretrain = optim.AdamW(teacher_model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        # 初始化预训练选择器（基于伪标签与结构稳定性，无标签泄露）
        pretrain_selector = PretrainBestSelector(
            pseudo_labels=y_pseudo_norm,
            top_k=50,
            min_epoch=max(10, pretrain_epochs // 10)  # 至少 10 epoch 后才开始选
        )
        print(f"Starting pretraining with {pretrain_epochs} epochs...")
        print(f"[Input Hardness] 动态特征掩码已启用，Mask列={CHEAT_INDICES}，概率=50%")
        
        for epoch in range(pretrain_epochs):
            if epoch == 0 or (epoch + 1) == pretrain_epochs:
                print(f"Pretrain epoch {epoch+1}/{pretrain_epochs}...")
            teacher_model.train()
            optimizer_pretrain.zero_grad()
            schedule = objective.get_loss_schedule(epoch, pretrain_epochs, phase="pretrain")
            
            # === 动态特征掩码 (Dynamic Feature Masking) ===
            # 1. 克隆特征，避免修改原始数据
            masked_features = features_t.clone()
            
            # 2. 50% 概率触发 Mask
            if np.random.rand() < 0.5:
                # 3. 列式抹除：将所有节点的指定特征列置为 0
                masked_features[:, CHEAT_INDICES] = 0.0
            
            # 4. 将 masked_features 传入模型，目标仍为融合伪标签
            loss, pred = teacher_model(masked_features, graph, y=y_pseudo_norm, phase="pretrain", loss_config=schedule)
            loss.backward()
            optimizer_pretrain.step()

            # 评估与选择最优 checkpoint（基于伪标签，无标签泄露）
            metrics = pretrain_selector.evaluate(pred, epoch + 1)
            if (epoch + 1) == pretrain_selector.best_epoch:
                pretrain_selector.update_best_state(teacher_model.state_dict())

            # 预训练阶段不调用全量 evaluate，避免昂贵评估
        
        # 预训练结束后加载最优 checkpoint（基于伪标签选出）
        if pretrain_selector.best_state is not None:
            teacher_model.load_state_dict(pretrain_selector.best_state)
            print(f"[Pretrain] Loaded best checkpoint from epoch {pretrain_selector.best_epoch}, "
                  f"score={pretrain_selector.best_score:.4f}")
    else:
        print(f"!!! [消融实验] 跳过预训练阶段 (run_tag={run_tag}) !!!")

    print("\n--- 阶段二：少量真实标签微调 ---")
    # 【重要】微调阶段使用增强的正则化强度 (FINETUNE_WEIGHT_DECAY)，防止在 1% 标签上过拟合
    optimizer_finetune = optim.AdamW(teacher_model.parameters(), lr=config.LR / 10, weight_decay=config.FINETUNE_WEIGHT_DECAY)
    best_score = -1.0
    best_model_weights = None
    # 记录微调阶段的历史最优指标，便于报告（避免过拟合错过最佳点）
    best_finetune_metrics = {
        'kendall': float('-inf'),
        'top20_jaccard': float('-inf'),
        'top50_jaccard': float('-inf'),
        'top100_jaccard': float('-inf'),
    }
    best_finetune_snapshot = None  # 保存完整指标快照
    best_finetune_points = {}      # 记录每个指标对应的 (epoch, value)
    print(f"Starting finetuning with {finetune_epochs} epochs...")
    for epoch in range(finetune_epochs):
        if epoch == 0 or (epoch + 1) == finetune_epochs:
            print(f"Finetune epoch {epoch+1}/{finetune_epochs}...")
        teacher_model.train()
        optimizer_finetune.zero_grad()
        global_epoch = pretrain_epochs + epoch
        schedule = objective.get_loss_schedule(epoch, finetune_epochs, phase="finetune")
        loss, _ = teacher_model(features_t, graph, y=y_real_norm, label_mask=label_mask, phase="finetune", loss_config=schedule)
        loss.backward()
        optimizer_finetune.step()

        if (epoch + 1) % 100 == 0 or epoch == finetune_epochs - 1:
            results = evaluate_model(teacher_model, graph, features_t, y_real_norm, y_min, y_max,
                                     eval_topk=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                     eval_ndcg=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            epoch_history.append(pretrain_epochs + epoch + 1)
            kendall_history.append(results['kendall'])
            top20_history.append(results['top20_jaccard'])
            top50_history.append(results['top50_jaccard'])
            top100_history.append(results['top100_jaccard'])

            # 精简日志，降低I/O开销
            if (epoch + 1) % 200 == 0 or epoch == finetune_epochs - 1:
                print(f"[Finetune Eval] ep={epoch+1}/{finetune_epochs} loss={loss.item():.4f} kendall={results['kendall']:.4f} top20={results['top20_jaccard']:.4f}")

            stage_num, _, _ = stage_manager.check_and_report_stage_change(pretrain_epochs + epoch, pretrain_epochs + finetune_epochs)
            detailed_score = calculate_combined_score_detailed(results, stage=stage_num)
            current_score = detailed_score["total_score"]
            if current_score > best_score:
                best_score = current_score
                best_model_weights = teacher_model.state_dict()
                print(f"  Saved best model. Combined Score: {best_score:.4f}")
            # 跟踪微调阶段的历史最优指标
            for key in best_finetune_metrics.keys():
                if results.get(key, float('-inf')) > best_finetune_metrics[key]:
                    best_finetune_metrics[key] = results[key]
                    best_finetune_snapshot = results
                    # 记录对应的全局 epoch（与图像 x 轴对齐）
                    best_finetune_points[key] = (pretrain_epochs + epoch + 1, results[key])

    # 微调阶段选择：基于真实标签的 Kendall（此阶段使用真实标签合理，因为是微调）
    if best_model_weights is not None:
        teacher_model.load_state_dict(best_model_weights)
        print(f"[Finetune] Loaded best checkpoint from finetuning phase")

    # 教师最终评估（仅含预训练版本）
    print("\n=== 教师模型最终评估 ===")
    teacher_results = evaluate_model(teacher_model, graph, features_t, y_real_norm, y_min, y_max,
                                     eval_topk=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                     eval_ndcg=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    epoch_history.append(pretrain_epochs + finetune_epochs)
    kendall_history.append(teacher_results['kendall'])
    top20_history.append(teacher_results['top20_jaccard'])
    top50_history.append(teacher_results['top50_jaccard'])
    top100_history.append(teacher_results['top100_jaccard'])
    if best_finetune_snapshot is None:
        best_finetune_snapshot = teacher_results
        best_finetune_metrics = {k: teacher_results.get(k, float('nan')) for k in best_finetune_metrics.keys()}
        best_finetune_points = {k: (pretrain_epochs + finetune_epochs, teacher_results.get(k, float('nan'))) for k in best_finetune_metrics.keys()}
    print(f"Teacher Results ({run_tag}):")
    print(f"  Kendall's: {teacher_results['kendall']:.4f}")
    print(f"  Spearman's ρ: {teacher_results['spearman']:.4f}")
    print(f"  NDCG@10: {teacher_results['ndcg_at_10']:.4f}, NDCG@20: {teacher_results['ndcg_at_20']:.4f}, NDCG@30: {teacher_results['ndcg_at_30']:.4f}")
    print(f"  Top-10 Accuracy: {teacher_results['top10_acc']:.4f}, Top-20 Accuracy: {teacher_results['top20_acc']:.4f}, Top-30 Accuracy: {teacher_results['top30_acc']:.4f}")
    print(f"  Top-10 Jaccard: {teacher_results['top10_jaccard']:.4f}, Top-20 Jaccard: {teacher_results['top20_jaccard']:.4f}, Top-30 Jaccard: {teacher_results['top30_jaccard']:.4f}")
    print(f"  Top-50 Jaccard: {teacher_results['top50_jaccard']:.4f}, Top-100 Jaccard: {teacher_results['top100_jaccard']:.4f}")
    print(f"  Imprecision@10: {teacher_results['imprecision_at_10']:.4f}, Imprecision@50: {teacher_results['imprecision_at_50']:.4f}, Imprecision@100: {teacher_results['imprecision_at_100']:.4f}")
    print(f"  MI: {teacher_results['mi']:.4f}")

    # # === 教师模型 Top-K 指标统计表 ===（已移至最后与学生模型一起输出）
    # print(f"\n" + "="*70)
    # print(f"Top-K 指标统计表（最佳教师模型）：")
    # unified_header = f"{'K':<6}|{'Top-K Jaccard':>16}|{'NDCG@K':>16}|{'Imprecision':>16}"
    # print(unified_header)
    # print('-' * len(unified_header))
    # 
    # for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    #     jaccard_key = f'top{k}_jaccard'
    #     ndcg_key = f'ndcg_at_{k}'
    #     imprecision_key = f'imprecision_at_{k}'
    #     
    #     jaccard_val = teacher_results.get(jaccard_key, float('nan'))
    #     ndcg_val = teacher_results.get(ndcg_key, float('nan'))
    #     imprecision_val = teacher_results.get(imprecision_key, float('nan'))
    #     
    #     print(f"{k:<6}|{jaccard_val:>16.6f}|{ndcg_val:>16.6f}|{imprecision_val:>16.6f}")
    # 
    # print("="*70)

    print("\n=== 测量教师模型推理时间 ===")
    teacher_inf_ms, teacher_inf_s = calculate_inference_time(teacher_model, graph, features_t, device, repetitions=100)
    print(f"  Teacher Inference Time: {teacher_inf_ms:.4f} ms ({teacher_inf_s:.6f} s)")
    teacher_inference_time = teacher_inf_s  # 保持向后兼容

    trend_path = None
    topk_trend_path = None

    return {
        "results": teacher_results,
        "inference_time": teacher_inference_time,
        "trend_path": trend_path,
        "topk_trend_path": topk_trend_path,
        "epoch_history": epoch_history,
        "kendall_history": kendall_history,
        "best_finetune_metrics": best_finetune_metrics,
        "best_finetune_snapshot": best_finetune_snapshot,
        "best_finetune_points": best_finetune_points,
        "model": teacher_model,
    }

    # print("\n=== [消融] 仅微调 (无预训练) ===")
    # without_pretrain = run_teacher_training(
    #     config,
    #     graph,
    #     features_t,
    #     y_real_norm,
    #     y_min,
    #     y_max,
    #     y_pseudo_norm,
    #     label_mask,
    #     enable_pretraining=False,
    #     run_tag="no_pretrain",
    # )

    # print("\n=== 结果对比 (含预训练 vs 无预训练) ===")
    # metrics = [
    #     ("kendall", "Kendall"),
    #     ("spearman", "Spearman"),
    #     ("ndcg_at_20", "NDCG@20"),
    #     ("top20_jaccard", "Top-20 Jaccard"),
    #     ("top50_jaccard", "Top-50 Jaccard"),
    #     ("top100_jaccard", "Top-100 Jaccard"),
    #     ("top20_acc", "Top-20 Acc"),
    # ]
    # header = f"{'Metric':<18}|{'With Pretrain':>14}|{'No Pretrain':>14}"
    # print(header)
    # print("-" * len(header))
    # for key, label in metrics:
    #     left = with_pretrain['results'].get(key, float('nan'))
    #     right = without_pretrain['results'].get(key, float('nan'))
    #     print(f"{label:<18}|{left:>14.4f}|{right:>14.4f}")

    # # 输出微调阶段历史最佳指标，防止过拟合掩盖最优点（已关闭）
    # print("\n=== 微调阶段历史最佳 (仅评估点) ===")
    # ...
    #     ("kendall", "Kendall"),
    #     ("spearman", "Spearman"),
    #     ("ndcg_at_20", "NDCG@20"),
    #     ("top20_jaccard", "Top-20 Jaccard"),
    #     ("top50_jaccard", "Top-50 Jaccard"),
    #     ("top100_jaccard", "Top-100 Jaccard"),
    #     ("top20_acc", "Top-20 Acc"),
    # ]
    # header = f"{'Metric':<18}|{'With Pretrain':>14}|{'No Pretrain':>14}"
    # print(header)
    # print("-" * len(header))
    # for key, label in metrics:
    #     left = with_pretrain['results'].get(key, float('nan'))
    #     right = without_pretrain['results'].get(key, float('nan'))
    #     print(f"{label:<18}|{left:>14.4f}|{right:>14.4f}")

    # # 输出微调阶段历史最佳指标，防止过拟合掩盖最优点
    # print("\n=== 微调阶段历史最佳 (仅评估点) ===")
    # finetune_best_keys = [
    #     ("kendall", "Kendall (best)"),
    #     ("top20_jaccard", "Top-20 Jaccard (best)"),
    #     ("top50_jaccard", "Top-50 Jaccard (best)"),
    #     ("top100_jaccard", "Top-100 Jaccard (best)"),
    # ]
    # header_best = f"{'Metric':<22}|{'With Pretrain':>14}|{'No Pretrain':>14}"
    # print(header_best)
    # print("-" * len(header_best))
    # for key, label in finetune_best_keys:
    #     left = with_pretrain['best_finetune_metrics'].get(key, float('nan'))
    #     right = without_pretrain['best_finetune_metrics'].get(key, float('nan'))
    #     print(f"{label:<22}|{left:>14.4f}|{right:>14.4f}")

    # return {
    #     "results": teacher_results,
    #     "inference_time": teacher_inference_time,
    #     "trend_path": trend_path,
    #     "topk_trend_path": topk_trend_path,
    #     "epoch_history": epoch_history,
    #     "kendall_history": kendall_history,
    #     "best_finetune_metrics": best_finetune_metrics,
    #     "best_finetune_snapshot": best_finetune_snapshot,
    #     "best_finetune_points": best_finetune_points,
    #     "model": teacher_model,
    # }


def train_student_with_kd(config, graph, features_t, y_real_norm, y_min, y_max, label_mask, teacher_model):
    device = features_t.device
    teacher_model.eval()
    with torch.no_grad():
        t_scores = teacher_model.predict(features_t, graph).squeeze(1)

    def build_student():
        if config.STUDENT_TYPE == "mlp":
            stu = StudentMLP(features_t.shape[1], hidden_dim=config.STUDENT_HIDDEN, dropout=config.STUDENT_DROPOUT).to(device)
            def forward_student():
                return stu(features_t).squeeze(1)
        elif config.STUDENT_TYPE == "sgc":
            stu = StudentSGCMLP(features_t.shape[1], hidden_dim=config.STUDENT_HIDDEN, dropout=config.STUDENT_DROPOUT, K=config.STUDENT_SGC_K).to(device)
            def forward_student():
                y, _ = stu(features_t, graph)
                return y.squeeze(1)
        elif config.STUDENT_TYPE == "sage":
            stu = StudentSAGEMLP(features_t.shape[1], hidden_dim=config.STUDENT_HIDDEN, dropout=config.STUDENT_DROPOUT, 
                               num_layers=config.STUDENT_SAGE_LAYERS, aggregator=config.STUDENT_SAGE_AGGREGATOR).to(device)
            def forward_student():
                y, _ = stu(features_t, graph)
                return y.squeeze(1)
        else:
            raise ValueError(f"Unknown student type: {config.STUDENT_TYPE}")
        return stu, forward_student

    student_model, forward_student = build_student()
    optimizer = optim.AdamW(student_model.parameters(), lr=0.005, weight_decay=1e-3)
    objective = RankingObjective(config)
    label_mask_bool = label_mask.squeeze(1).bool()

    best = {"score": -1e9, "state": None, "metrics": None, "criterion": config.STUDENT_SAVE_BY}
    
    # 初始化历史最佳值追踪（用于统计最佳结果，不仅仅是Kendall最佳时刻）
    best_metrics_history = {}
    # 基础指标
    best_metrics_history['kendall'] = -1.0
    best_metrics_history['spearman'] = -1.0
    best_metrics_history['mi'] = -1.0
    # Top-K指标
    for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        best_metrics_history[f'top{k}_jaccard'] = -1.0
        best_metrics_history[f'ndcg_at_{k}'] = -1.0
        best_metrics_history[f'imprecision_at_{k}'] = -1.0
    
    max_epochs = 300
    
    # 记录训练历史
    epoch_history = []
    kendall_history = []
    top10_history = []
    top50_history = []
    top100_history = []

    for ep in range(1, max_epochs + 1):
        student_model.train()
        optimizer.zero_grad()
        s_scores = forward_student()

        idx_i_g, idx_j_g = sample_pairs_global(s_scores.numel(), config.KD_NUM_PAIRS, device)
        loss_g = kd_pairwise_bce(s_scores, t_scores, idx_i_g, idx_j_g, T=config.KD_T,
                                 use_margin_weight=config.KD_USE_MARGIN_WEIGHT,
                                 margin_clip=config.KD_MARGIN_CLIP)

        idx_i_k, idx_j_k = sample_pairs_topk(t_scores, config.KD_TOPK, config.KD_NUM_PAIRS // 2, device)
        loss_k = kd_pairwise_bce(s_scores, t_scores, idx_i_k, idx_j_k, T=config.KD_T,
                                 use_margin_weight=config.KD_USE_MARGIN_WEIGHT,
                                 margin_clip=config.KD_MARGIN_CLIP)

        loss_list = kd_listwise_kl(s_scores, t_scores, T=config.KD_T) if config.KD_LAMBDA_LIST > 0 else 0.0

        if label_mask_bool.any():
            if config.KD_GT_USE_RANKLOSS:
                schedule = config.TEACHER_LOSS_CONFIG.copy()
                schedule["hard_negatives"] = False
                schedule["sample_size"] = min(schedule.get("sample_size", 50000), 20000)
                loss_gt = objective.compute_loss(s_scores.unsqueeze(1), y_real_norm, label_mask, schedule)
            else:
                loss_gt = F.mse_loss(s_scores[label_mask_bool], y_real_norm[label_mask_bool].view(-1))
        else:
            loss_gt = torch.tensor(0.0, device=device)

        loss = (
            config.KD_ALPHA_GLOBAL * loss_g
            + config.KD_ALPHA_TOPK * loss_k
            + (config.KD_LAMBDA_LIST * loss_list if isinstance(loss_list, torch.Tensor) else config.KD_LAMBDA_LIST * torch.tensor(loss_list, device=device))
            + config.KD_LAMBDA_GT * loss_gt
        )

        loss.backward()
        optimizer.step()

        if ep % 50 == 0 or ep == max_epochs:
            student_model.eval()
            metrics = evaluate_model(student_model, graph, features_t, y_real_norm, y_min, y_max,
                                     eval_topk=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                     eval_ndcg=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            kendall_val = metrics.get("kendall", -1.0)
            top10_val = metrics.get("top10_jaccard", -1.0)
            top50_val = metrics.get("top50_jaccard", -1.0)
            top100_val = metrics.get("top100_jaccard", -1.0)
            
            # 记录历史
            epoch_history.append(ep)
            kendall_history.append(kendall_val)
            top10_history.append(top10_val)
            top50_history.append(top50_val)
            top100_history.append(top100_val)
            
            # 根据 STUDENT_SAVE_BY 参数动态选择评估标准
            if config.STUDENT_SAVE_BY == "kendall":
                current_score = kendall_val
            elif config.STUDENT_SAVE_BY == "topk":
                # 使用 Top-10/50/100 Jaccard 的平均值作为 Top-K 标准
                current_score = (top10_val + top50_val + top100_val) / 3.0
            elif config.STUDENT_SAVE_BY == "constraint":
                # 组合标准：70% Kendall + 30% Top-10/50/100 Jaccard 平均值
                current_score = kendall_val * 0.7 + (top10_val + top50_val + top100_val) / 3.0 * 0.3
            else:
                # 默认使用 Kendall
                current_score = kendall_val
            
            if current_score > best["score"]:
                best["score"] = current_score
                best["state"] = deepcopy(student_model.state_dict())
                best["metrics"] = metrics
            
            # 更新历史最佳值（跟踪每个k的最佳结果）
            # 基础指标
            best_metrics_history['kendall'] = max(best_metrics_history['kendall'], metrics.get('kendall', -1.0))
            best_metrics_history['spearman'] = max(best_metrics_history['spearman'], metrics.get('spearman', -1.0))
            best_metrics_history['mi'] = max(best_metrics_history['mi'], metrics.get('mi', -1.0))
            
            # Top-K指标
            for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                jaccard_key = f'top{k}_jaccard'
                ndcg_key = f'ndcg_at_{k}'
                imprecision_key = f'imprecision_at_{k}'
                
                if jaccard_key in metrics:
                    best_metrics_history[jaccard_key] = max(best_metrics_history[jaccard_key], metrics[jaccard_key])
                if ndcg_key in metrics:
                    best_metrics_history[ndcg_key] = max(best_metrics_history[ndcg_key], metrics[ndcg_key])
                if imprecision_key in metrics:
                    # Imprecision 越小越好，所以取最小值
                    if best_metrics_history[imprecision_key] < 0:
                        best_metrics_history[imprecision_key] = metrics[imprecision_key]
                    else:
                        best_metrics_history[imprecision_key] = min(best_metrics_history[imprecision_key], metrics[imprecision_key])
            if ep % 100 == 0 or ep == max_epochs:
                safe_loss = loss.detach().item() if hasattr(loss, "detach") else float(loss)
                print(f"[Student-{config.STUDENT_TYPE}] ep={ep} loss={safe_loss:.4f} kendall={kendall_val:.4f} top50={top50_val:.4f}")

    if best["state"] is not None:
        student_model.load_state_dict(best["state"])
        final_metrics = best["metrics"]
    else:
        student_model.eval()
        final_metrics = evaluate_model(student_model, graph, features_t, y_real_norm, y_min, y_max,
                                       eval_topk=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                       eval_ndcg=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    return {
        "model": student_model,
        "metrics": final_metrics,
        "best_score": best["score"],
        "best_criterion": best["criterion"],
        "best_metrics_history": best_metrics_history,
        "epoch_history": epoch_history,
        "kendall_history": kendall_history,
        "top10_history": top10_history,
        "top50_history": top50_history,
        "top100_history": top100_history,
    }


def main(config: Config):
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {config.DATASET_NAME}")

    import utils_data
    print("Step 1: Loading data...")
    data_path = os.path.join("data", f"{config.DATASET_NAME}.txt")
    start_time = time.time()
    adj, features, nodes = utils_data.load_data(
        data_path,
        log_transform=config.LOG_TRANSFORM,
        normalize_features=config.NORMALIZE_FEATURES,
    )
    end_time = time.time()
    print(f"Data loading completed in {end_time - start_time:.2f} seconds")
    print(f"Number of nodes: {adj.shape[0]}")
    print(f"Number of features: {features.shape[1]}")
    print(f"Adjacency matrix shape: {adj.shape}")

    # 先保留完整特征用于直接读取伪标签，再筛选训练特征
    full_features = features.copy()

    # 特征筛选：保留更稳定的指标组合（与源脚本一致）

    selected_features = [0, 1, 2, 4, 7, 8, 10, 11, 12]
    # selected_features = [0, 4, 7, 8, 10, 11, 12]

    features = features[:, selected_features]
    print(f"特征筛选完成，保留{len(selected_features)}个指标，特征形状: {features.shape}")

    sir_path = utils_data.get_sir_path(config.DATASET_NAME, config.PROPAGATION_THRESHOLD_MULTIPLE)
    if not os.path.exists(sir_path):
        print(f"Error: SIR file not found at {sir_path}")
        sir_dir = os.path.dirname(sir_path)
        if os.path.exists(sir_dir):
            available_files = os.listdir(sir_dir)
            print(f"Available SIR files in {sir_dir}:")
            for f in available_files:
                print(f"  {f}")
        raise FileNotFoundError("SIR文件缺失")
    sir_dict = utils_data.load_sir_scores(sir_path)
    y_real = np.array([sir_dict[int(node)] for node in nodes])

    # === 修改步骤 1: 生成"融合伪标签" (Fusion Pseudo-labels - Target Alignment) ===
    print("正在生成融合伪标签 (Fusion Pseudo-labels)...")
    print(f"伪标签生成方法: {config.PSEUDO_LABEL_METHOD}")
    
    # 特征列的映射索引
    # [0:Degree, 1:Betweenness, 2:Closeness, 3:Eigenvector, 4:PageRank, 5:ClusteringCoeff,
    #  6:AvgNeighborDegree, 7:KShell, 8:HIndex, 9-12:LDP...]
    idx_map = {
        'degree': 0,
        'pr': 4,
        'clustering_coeff': 5,
        'kshell': 7,
        'hindex': 8
    }
    
    # 根据配置选择指标
    if config.PSEUDO_LABEL_METHOD == "degree":
        selected_metrics = ['degree']
    elif config.PSEUDO_LABEL_METHOD == "degree_cc_hindex":
        selected_metrics = ['degree', 'clustering_coeff', 'hindex']
    elif config.PSEUDO_LABEL_METHOD == "degree_kshell_hindex_pr":
        selected_metrics = ['degree', 'kshell', 'hindex', 'pr']
    else:
        selected_metrics = ['degree', 'clustering_coeff', 'hindex']  # 默认
    
    # 计算融合分数 (使用 Rank Normalization 消除量纲差异)
    from scipy.stats import rankdata
    
    fusion_score = np.zeros(full_features.shape[0])
    
    for metric in selected_metrics:
        col_idx = idx_map[metric]
        raw_values = full_features[:, col_idx]  # 获取该指标的整列原始值
        
        # 转换为百分比排名 (0.0 ~ 1.0)，消除量纲差异
        rank_norm = rankdata(raw_values, method='average') / len(raw_values)
        fusion_score += rank_norm  # 累加排名
    
    # 取平均值作为最终伪标签
    y_pseudo = fusion_score / len(selected_metrics)
    
    # Kendall 相关性评估
    try:
        kendall_corr, _ = kendalltau(y_pseudo, y_real)
    except:
        kendall_corr = -1.0
    
    print(f"融合指标完成，使用了: {selected_metrics}")
    print(f"伪标签与真实SIR的Kendall相关性: {kendall_corr:.4f}")
    print(f"Pseudo-labels stats: min={np.min(y_pseudo):.4f}, max={np.max(y_pseudo):.4f}, mean={np.mean(y_pseudo):.4f}, std={np.std(y_pseudo):.4f}")

    # 将邻接矩阵转换为 PyG 的 edge_index / edge_weight
    if isinstance(adj, np.ndarray):
        adj = sparse.csr_matrix(adj)
    adj_coo = adj.tocoo()
    edge_index = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
    edge_weight = torch.FloatTensor(adj_coo.data) if adj_coo.data is not None else None
    graph = {
        'edge_index': edge_index.to(device),
        'edge_weight': edge_weight.to(device) if edge_weight is not None else None,
    }
    features_t = torch.FloatTensor(features).to(device)
    y_real_t = torch.FloatTensor(y_real).unsqueeze(1).to(device)
    y_real_norm, y_min, y_max = normalize_labels(y_real_t)
    y_pseudo_t = torch.FloatTensor(y_pseudo).unsqueeze(1).to(device)
    y_pseudo_norm, _, _ = normalize_labels(y_pseudo_t)

    y_few, label_mask = generate_few_labels(
        y_real_norm,
        ratio=config.FEW_LABELS_RATIO,
        selection_strategy=config.FEW_LABELS_STRATEGY,
        config=config,
        y_pseudo=y_pseudo_norm,
        labels_num=config.LABELS_NUM,
    )

    # === 定义教师模型权重保存路径 ===
    teacher_weights_path = f"checkpoints/teacher_{config.DATASET_NAME}_{config.TEACHER_GNN_TYPE}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    # 训练峰值内存监控（EGKD：教师训练 + 学生蒸馏）
    mem_tracker = PeakMemoryTracker(interval_sec=0.05)
    mem_tracker.start()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # === 逻辑分支：加载现有模型 OR 重新训练 ===
    # 如果想强制重新训练，设置 FORCE_RETRAIN = True
    FORCE_RETRAIN = True

    teacher_model = None
    with_pretrain = None

    if os.path.exists(teacher_weights_path) and not FORCE_RETRAIN:
        print(f"\n[Info] 发现已保存的教师模型权重: {teacher_weights_path}，正在加载...")
        # 初始化模型结构 (必须与训练时参数一致)
        teacher_model = RankingGNNModel(
            in_feats=features_t.shape[1],
            hidden_dim=config.TEACHER_HIDDEN_DIM,
            num_layers=config.TEACHER_GNN_LAYERS,
            dropout=config.TEACHER_DROPOUT,
            gnn_type=config.TEACHER_GNN_TYPE,
            gat_heads=config.TEACHER_GAT_HEADS,
            sage_aggregator=config.TEACHER_SAGE_AGGREGATOR,
            sgc_k=config.TEACHER_SGC_K,
        ).to(device)

        # 加载参数
        teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device))
        teacher_model.eval()

        # 补全 with_pretrain 字典结构，以便后续代码兼容
        # 注意：使用与训练时相同的 eval_topk 和 eval_ndcg 范围，避免 nan 值
        eval_res = evaluate_model(teacher_model, graph, features_t, y_real_norm, y_min, y_max,
                                  eval_topk=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                  eval_ndcg=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        # 计算推理时间
        print("\n=== 测量教师模型推理时间 ===")
        teacher_inf_ms, teacher_inf_s = calculate_inference_time(teacher_model, graph, features_t, device, repetitions=100)
        
        with_pretrain = {
            'model': teacher_model,
            'results': eval_res,
            'inference_time': teacher_inf_s,
            'inference_time_ms': teacher_inf_ms,
        }
        print(f"教师模型加载完成。Kendall={eval_res.get('kendall', float('nan')):.4f}")
        print(f"教师模型推理时间: {teacher_inf_ms:.4f} ms ({teacher_inf_s:.6f} s)")

    else:
        # === 原有的训练流程 ===
        print("\n=== 运行含预训练版本 ===")
        # 种子重置：教师模型训练前
        set_seed(config.SEED)
        with_pretrain = run_teacher_training(
            config,
            graph,
            features_t,
            y_real_norm,
            y_min,
            y_max,
            y_pseudo_norm,
            label_mask,
            enable_pretraining=True,
            run_tag="with_pretrain",
        )

        # === 新增：保存模型权重 ===
        print(f"[Info] 保存教师模型权重至: {teacher_weights_path}")
        torch.save(with_pretrain['model'].state_dict(), teacher_weights_path)

    # print("\n=== [消融] 仅微调 (无预训练) ===")
    # without_pretrain = run_teacher_training(
    #     config,
    #     graph,
    #     features_t,
    #     y_real_norm,
    #     y_min,
    #     y_max,
    #     y_pseudo_norm,
    #     label_mask,
    #     enable_pretraining=False,
    #     run_tag="no_pretrain",
    # )

    # # 结果对比 (含预训练 vs 无预训练) — 已停用
    # metrics = [
    #     ("kendall", "Kendall"),
    #     ("spearman", "Spearman"),
    #     ("ndcg_at_20", "NDCG@20"),
    #     ("top20_jaccard", "Top-20 Jaccard"),
    #     ("top50_jaccard", "Top-50 Jaccard"),
    #     ("top100_jaccard", "Top-100 Jaccard"),
    #     ("top20_acc", "Top-20 Acc"),
    # ]
    # header = f"{'Metric':<18}|{'With Pretrain':>14}|{'No Pretrain':>14}"
    # print(header)
    # print("-" * len(header))
    # for key, label in metrics:
    #     left = with_pretrain['results'].get(key, float('nan'))
    #     right = without_pretrain['results'].get(key, float('nan'))
    #     print(f"{label:<18}|{left:>14.4f}|{right:>14.4f}")

    # # 微调阶段历史最佳 (仅评估点) — 已停用
    # finetune_best_keys = [
    #     ("kendall", "Kendall (best)"),
    #     ("top20_jaccard", "Top-20 Jaccard (best)"),
    #     ("top50_jaccard", "Top-50 Jaccard (best)"),
    #     ("top100_jaccard", "Top-100 Jaccard (best)"),
    # ]
    # header_best = f"{'Metric':<22}|{'With Pretrain':>14}|{'No Pretrain':>14}"
    # print(header_best)
    # print("-" * len(header_best))
    # for key, label in finetune_best_keys:
    #     left = with_pretrain['best_finetune_metrics'].get(key, float('nan'))
    #     right = without_pretrain['best_finetune_metrics'].get(key, float('nan'))
    #     print(f"{label:<22}|{left:>14.4f}|{right:>14.4f}")

    # === 学生蒸馏训练（仅保留 SAGE 学生模型） ===
    if config.STUDENT_ENABLE:
        teacher_for_kd = with_pretrain.get('model', None)
        if teacher_for_kd is None:
            raise RuntimeError("Teacher model not available for KD")

        student_results = {}
        config.STUDENT_TYPE = "sage"
        print("\n=== 训练学生模型 (sage)：KD + 少量真实标签校准 ===")
        print(f"最佳模型选择标准: {config.STUDENT_SAVE_BY}")
        # 种子重置：学生模型训练前
        set_seed(config.SEED)
        stu_out = train_student_with_kd(config, graph, features_t, y_real_norm, y_min, y_max, label_mask, teacher_for_kd)
        student_results['sage'] = stu_out
        m = stu_out['metrics']
        print(f"[Student-sage] Kendall={m.get('kendall', float('nan')):.4f} | Top20J={m.get('top20_jaccard', float('nan')):.4f} | Top50J={m.get('top50_jaccard', float('nan')):.4f} | Top100J={m.get('top100_jaccard', float('nan')):.4f}")
        print(f"最佳模型得分 ({stu_out['best_criterion']}): {stu_out['best_score']:.4f}")

        # 计算学生模型推理时间
        print("\n=== 测量学生模型推理时间 ===")
        sage_inf_ms, sage_inf_s = calculate_inference_time(stu_out['model'], graph, features_t, device, repetitions=100)
        student_results['sage']['inference_time'] = sage_inf_s
        student_results['sage']['inference_time_ms'] = sage_inf_ms
        print(f"学生模型推理时间: {sage_inf_ms:.4f} ms ({sage_inf_s:.6f} s)")

        # 教师 vs 学生（SAGE）指标对比（含增幅、NDCG@10/50/100）
        print("\n=== 教师 vs 学生(SAGE) 指标对比（含增幅） ===")
        teacher_metrics = with_pretrain['results']
        teacher_inf = with_pretrain.get('inference_time', float('nan'))

        def pct(delta_num, base):
            if base is None or np.isnan(base) or abs(base) < 1e-8:
                return float('nan')
            return (delta_num / base) * 100.0

        def val_sage(key):
            return student_results.get('sage', {}).get('metrics', {}).get(key, float('nan'))

        metrics_rows = [
            ("Kendall", "kendall"),
            ("Spearman", "spearman"),
            ("MI", "mi"),
            ("Top-10 Jaccard", "top10_jaccard"),
            ("Top-50 Jaccard", "top50_jaccard"),
            ("Top-100 Jaccard", "top100_jaccard"),
            ("NDCG@10", "ndcg_at_10"),
            ("NDCG@50", "ndcg_at_50"),
            ("NDCG@100", "ndcg_at_100"),
            ("Imprecision@10", "imprecision_at_10"),
            ("Imprecision@50", "imprecision_at_50"),
            ("Imprecision@100", "imprecision_at_100"),
        ]

        header2 = f"{'Metric':<18}|{'Teacher':>12}|{'SAGE':>12}|{'Δ SAGE%':>10}"
        print(header2)
        print('-' * len(header2))
        for name, key in metrics_rows:
            t_val = teacher_metrics.get(key, float('nan'))
            s_val = val_sage(key)
            d_val = pct(s_val - t_val, t_val)
            print(f"{name:<18}|{t_val:>12.4f}|{s_val:>12.4f}|{d_val:>10.2f}")

        # === 推理时间对比（详细表格展示）===
        print("\n" + "="*70)
        print("=== 教师 vs 学生 推理时间对比（Knowledge Distillation 关键指标）===")
        print("="*70)
        
        teacher_inf_s = with_pretrain.get('inference_time', float('nan'))
        teacher_inf_ms = with_pretrain.get('inference_time_ms', teacher_inf_s * 1000)
        sage_inf_s = student_results['sage'].get('inference_time', float('nan'))
        sage_inf_ms = student_results['sage'].get('inference_time_ms', sage_inf_s * 1000)
        
        # 计算加速比
        if not np.isnan(teacher_inf_s) and not np.isnan(sage_inf_s) and sage_inf_s > 0:
            speedup = teacher_inf_s / sage_inf_s
        else:
            speedup = float('nan')
        
        # 详细表格
        inf_header = f"{'Model':<15}|{'Time (ms)':>12}|{'Time (s)':>12}|{'Speedup':>10}"
        print(inf_header)
        print('-' * len(inf_header))
        print(f"{'Teacher (GNN)':<15}|{teacher_inf_ms:>12.4f}|{teacher_inf_s:>12.6f}|{'1.00x':>10}")
        print(f"{'Student (SAGE)':<15}|{sage_inf_ms:>12.4f}|{sage_inf_s:>12.6f}|{speedup:>9.2f}x")
        print('-' * len(inf_header))
        
        # 关键洞察
        if not np.isnan(speedup):
            if speedup > 1:
                percentage = ((speedup - 1) * 100)
                print(f"\n✓ 学生模型比教师模型快 {speedup:.2f}x（提升 {percentage:.1f}%）")
            elif speedup < 1:
                slowdown = 1 / speedup
                percentage = ((slowdown - 1) * 100)
                print(f"\n✗ 学生模型比教师模型慢 {slowdown:.2f}x（降低 {percentage:.1f}%）")
            else:
                print(f"\n= 推理时间相同")
        
        print("="*70)

        # === 学生模型详细指标输出（历史最佳值） ===
        print("\n" + "="*90)
        print("=== 学生模型（SAGE）Top-K 指标统计 - 历史最佳值 ===")
        print("="*90)
        sage_metrics = student_results['sage']['metrics']
        best_metrics_history = student_results['sage'].get('best_metrics_history', {})
        
        # 基础指标（当前/最终值）
        print(f"\n基础指标（最终轮次）：")
        print(f"  Kendall Tau Coefficient: {sage_metrics.get('kendall', float('nan')):.6f}  (历史最佳: {best_metrics_history.get('kendall', float('nan')):.6f})")
        print(f"  Spearman Correlation:    {sage_metrics.get('spearman', float('nan')):.6f}  (历史最佳: {best_metrics_history.get('spearman', float('nan')):.6f})")
        print(f"  Monotonicity Index (MI): {sage_metrics.get('mi', float('nan')):.6f}  (历史最佳: {best_metrics_history.get('mi', float('nan')):.6f})")
        
        # # 统一的Top-K指标表格（历史最佳值）
        # print(f"\nTop-K 指标统计表（历史最佳值）：")
        # unified_header = f"{'K':<6}|{'Top-K Jaccard':>16}|{'NDCG@K':>16}|{'Imprecision':>16}"
        # print(unified_header)
        # print('-' * len(unified_header))
        # 
        # for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        #     jaccard_key = f'top{k}_jaccard'
        #     ndcg_key = f'ndcg_at_{k}'
        #     imprecision_key = f'imprecision_at_{k}'
        #     
        #     jaccard_val = best_metrics_history.get(jaccard_key, float('nan'))
        #     ndcg_val = best_metrics_history.get(ndcg_key, float('nan'))
        #     imprecision_val = best_metrics_history.get(imprecision_key, float('nan'))
        #     
        #     print(f"{k:<6}|{jaccard_val:>16.6f}|{ndcg_val:>16.6f}|{imprecision_val:>16.6f}")
        # 
        # print("="*90)

        # === Top-K 指标统计表（最佳学习模型的最终预测结果） ===
        print("\n" + "="*90)
        print("=== 学生模型（SAGE）Top-K 指标统计 - 最佳学生模型 ===")
        print("="*90)
        
        print(f"\nTop-K 指标统计表（最佳学习模型）：")
        unified_header = f"{'K':<6}|{'Top-K Jaccard':>16}|{'NDCG@K':>16}|{'Imprecision':>16}"
        print(unified_header)
        print('-' * len(unified_header))
        
        for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            jaccard_key = f'top{k}_jaccard'
            ndcg_key = f'ndcg_at_{k}'
            imprecision_key = f'imprecision_at_{k}'
            
            jaccard_val = sage_metrics.get(jaccard_key, float('nan'))
            ndcg_val = sage_metrics.get(ndcg_key, float('nan'))
            imprecision_val = sage_metrics.get(imprecision_key, float('nan'))
            
            print(f"{k:<6}|{jaccard_val:>16.6f}|{ndcg_val:>16.6f}|{imprecision_val:>16.6f}")
        
        print("="*90)

        # === Top-K 指标统计表（教师模型对比） ===
        print("\n" + "="*90)
        print("=== 教师模型（SAGE）Top-K 指标统计 - 最佳教师模型 ===")  
        print("="*90)
        
        teacher_metrics = with_pretrain.get('results', {})
        print(f"\nTop-K 指标统计表（最佳教师模型）：")
        unified_header = f"{'K':<6}|{'Top-K Jaccard':>16}|{'NDCG@K':>16}|{'Imprecision':>16}"
        print(unified_header)
        print('-' * len(unified_header))
        
        for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            jaccard_key = f'top{k}_jaccard'
            ndcg_key = f'ndcg_at_{k}'
            imprecision_key = f'imprecision_at_{k}'
            
            jaccard_val = teacher_metrics.get(jaccard_key, float('nan'))
            ndcg_val = teacher_metrics.get(ndcg_key, float('nan'))
            imprecision_val = teacher_metrics.get(imprecision_key, float('nan'))
            
            print(f"{k:<6}|{jaccard_val:>16.6f}|{ndcg_val:>16.6f}|{imprecision_val:>16.6f}")
        
        print("="*90)

        # === 生成学生模型 Top-K 趋势可视化 ===（已注释）
        # print("\n=== 生成学生模型 Top-K 趋势可视化 ===")
        # sage_output = student_results['sage']
        # plot_student_topk_trends_save(
        #     student_model_type='sage',
        #     epoch_history=sage_output.get('epoch_history', []),
        #     top10_history=sage_output.get('top10_history', []),
        #     top50_history=sage_output.get('top50_history', []),
        #     top100_history=sage_output.get('top100_history', []),
        #     dataset_name=config.DATASET_NAME
        # )

        # === 教师 vs 学生 历史最佳肯德尔系数对比表（按需启用） ===
        # 下方输出已关闭，若需要可取消注释
        # print("\n" + "="*90)
        # print("=== 教师 vs 学生 历史最佳肯德尔系数对比 ===")
        # print("="*90)
        # teacher_final_kendall = with_pretrain['results'].get('kendall', float('nan'))
        # best_student_dict = student_results['sage'].get('best', {})
        # student_best_kendall = best_student_dict.get('metrics', {}).get('kendall', float('nan'))
        # if pd.isna(student_best_kendall):
        #     student_best_kendall = student_results['sage']['best_metrics_history'].get('kendall', float('nan'))
        # if not (pd.isna(teacher_final_kendall) or pd.isna(student_best_kendall)):
        #     kendall_diff = student_best_kendall - teacher_final_kendall
        #     kendall_retention = (student_best_kendall / teacher_final_kendall * 100) if teacher_final_kendall != 0 else 0
        # else:
        #     kendall_diff = float('nan')
        #     kendall_retention = float('nan')
        # print(f"\n{'指标':<25} | {'教师模型':<15} | {'学生模型':<15} | {'差值':<15} | {'保留率':<10}")
        # print("-" * 80)
        # print(f"{'最佳模型肯德尔系数':<25} | {teacher_final_kendall:<15.6f} | {student_best_kendall:<15.6f} | {kendall_diff:<15.6f} | {kendall_retention:<10.2f}%")
        # print("="*90)

    # === 返回关键指标（用于批量实验） ===
    if device.type == 'cuda':
        torch.cuda.synchronize()
    cpu_peak_mb = mem_tracker.stop()
    gpu_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == 'cuda' else 0.0
    gpu_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 / 1024 if device.type == 'cuda' else 0.0

    print("\n=== EGKD 训练峰值内存统计 ===")
    print(f"CPU RSS峰值: {cpu_peak_mb:.2f} MB")
    if device.type == 'cuda':
        print(f"GPU峰值已分配显存: {gpu_peak_allocated_mb:.2f} MB")
        print(f"GPU峰值保留显存: {gpu_peak_reserved_mb:.2f} MB")
    else:
        print("GPU峰值显存: N/A (当前为CPU运行)")

    return_results = {
        'teacher': {
            'kendall': with_pretrain['results'].get('kendall', float('nan')),
            'spearman': with_pretrain['results'].get('spearman', float('nan')),
            'inference_time_ms': with_pretrain.get('inference_time_ms', float('nan')),
            'top10_jaccard': with_pretrain['results'].get('top10_jaccard', float('nan')),
            'top50_jaccard': with_pretrain['results'].get('top50_jaccard', float('nan')),
            'top100_jaccard': with_pretrain['results'].get('top100_jaccard', float('nan')),
        }
    }
    
    if config.STUDENT_ENABLE and 'sage' in student_results:
        return_results['student'] = {
            'kendall': student_results['sage']['metrics'].get('kendall', float('nan')),
            'spearman': student_results['sage']['metrics'].get('spearman', float('nan')),
            'inference_time_ms': student_results['sage'].get('inference_time_ms', float('nan')),
            'top10_jaccard': student_results['sage']['metrics'].get('top10_jaccard', float('nan')),
            'top50_jaccard': student_results['sage']['metrics'].get('top50_jaccard', float('nan')),
            'top100_jaccard': student_results['sage']['metrics'].get('top100_jaccard', float('nan')),
        }

    return_results['memory_profile'] = {
        'cpu_peak_rss_mb': cpu_peak_mb,
        'gpu_peak_allocated_mb': gpu_peak_allocated_mb,
        'gpu_peak_reserved_mb': gpu_peak_reserved_mb,
        'device': str(device),
        'dataset': config.DATASET_NAME,
    }
    
    return return_results

    #     # 绘制学生模型的训练趋势可视化（Kendall 与 Top-K）
    #     print("\n生成学生模型训练趋势可视化...")
    #     plot_student_kendall_trends(student_results)
    #     plot_student_topk_trends(student_results)

    # # 合并可视化：将含预训练与无预训练曲线放在同一张图，横坐标按总训练轮次对齐
    # compare_trend_path = f"kendall_trend_compare_{config.DATASET_NAME}.png"
    # plot_kendall_trend_compare(
    #     with_pretrain['epoch_history'],
    #     with_pretrain['kendall_history'],
    #     without_pretrain['epoch_history'],
    #     without_pretrain['kendall_history'],
    #     pretrain_epochs=config.PRETRAIN_EPOCHS,
    #     best_with={"epoch": with_pretrain['best_finetune_points']['kendall'][0], "value": with_pretrain['best_finetune_points']['kendall'][1]} if 'best_finetune_points' in with_pretrain and 'kendall' in with_pretrain['best_finetune_points'] else None,
    #     best_without={"epoch": without_pretrain['best_finetune_points']['kendall'][0], "value": without_pretrain['best_finetune_points']['kendall'][1]} if 'best_finetune_points' in without_pretrain and 'kendall' in without_pretrain['best_finetune_points'] else None,
    #     save_path=compare_trend_path,
    # )
    # print(f"Kendall 对比曲线已保存至: {compare_trend_path}")

    # print("\n曲线文件:")
    # print(f"  含预训练: {with_pretrain['trend_path']}")
    # print(f"  无预训练: {without_pretrain['trend_path']}")


if __name__ == "__main__":
    cfg = Config()
    # cfg.DATASET_NAME = "Hamster"
    main(cfg)
