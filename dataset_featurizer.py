"""多模态分子任务的数据特征工程模块。

该模块负责：
1. 图结构特征（节点/边）构建；
2. 分子描述符与指纹特征提取；
3. 训练数据集对象封装。
"""

import random
import warnings

import numpy as np
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors, rdchem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

# 指纹配置
MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048
ATOM_PAIR_NUM_BITS = 2048
MACCS_NUM_BITS = 167

warnings.filterwarnings("ignore")
rdBase.DisableLog("rdApp.warning")


def seed_everything(seed: int = 42) -> None:
    """固定随机种子，尽可能保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)


def one_hot_encoding(value, choices):
    """对离散特征做 one-hot，额外保留 unknown 桶。"""
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


class MoleculeFeaturizer:
    """将 RDKit Mol 转为 PyG 图特征。"""

    def _atom_featurizer(self, atom):
        # 37种原子 + 1种其他
        atomic_number = [
            1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20,
            21, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 47, 48,
            49, 50, 51, 52, 53,
        ]
        return (
            one_hot_encoding(atom.GetAtomicNum(), atomic_number)
            + one_hot_encoding(atom.GetTotalDegree(), list(range(5)))
            + one_hot_encoding(
                int(atom.GetHybridization()),
                list(range(len(Chem.HybridizationType.names) - 1)),
            )
            + one_hot_encoding(
                atom.GetChiralTag(),
                list(range(len(Chem.ChiralType.names) - 1)),
            )
            + one_hot_encoding(atom.GetTotalNumHs(), list(range(5)))
            + [1 if atom.GetIsAromatic() else 0]
        )

    def _bond_featurizer(self, bond):
        bond_type = [
            int(bond.GetBondType() == rdchem.BondType.SINGLE),
            int(bond.GetBondType() == rdchem.BondType.DOUBLE),
            int(bond.GetBondType() == rdchem.BondType.TRIPLE),
            int(bond.GetBondType() == rdchem.BondType.AROMATIC),
        ]
        return bond_type + [int(bond.GetIsConjugated()), int(bond.IsInRing())]

    def __call__(self, mol):
        atom_features = [self._atom_featurizer(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float32)

        adj = Chem.GetAdjacencyMatrix(mol)
        coo_adj = coo_matrix(adj)
        row, col = coo_adj.row, coo_adj.col
        edge_index = torch.tensor([row, col], dtype=torch.long)

        bond_features = []
        for i, j in zip(row, col):
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            bond_features.append(self._bond_featurizer(bond))

        if len(bond_features) > 0:
            edge_attr = torch.tensor(bond_features, dtype=torch.float32)
        else:
            edge_attr = torch.empty((0, 6), dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_descriptors(mol):
    """提取常用 9 种分子描述符。"""
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.RingCount(mol),
    ]


def _bitvect_to_numpy(bit_vect, size: int) -> np.ndarray:
    """将 RDKit ExplicitBitVect 转为固定长度 numpy 向量。"""
    arr = np.zeros((size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vect, arr)
    return arr


def _sparse_fp_to_numpy(sparse_fp, size: int) -> np.ndarray:
    """将 RDKit 稀疏指纹转为固定长度 dense 向量。"""
    arr = np.zeros((size,), dtype=np.float32)
    for idx, value in sparse_fp.GetNonzeroElements().items():
        if 0 <= idx < size:
            arr[idx] = float(value)
    return arr


def get_fingerprint(mol):
    """拼接多种指纹为统一长度特征向量。"""
    try:
        # 1) Hashed Atom Pair (counts)
        fp_atom_pairs = rdMolDescriptors.GetHashedAtomPairFingerprint(
            mol, nBits=ATOM_PAIR_NUM_BITS, use2D=True
        )
        fp_atom_pairs = _sparse_fp_to_numpy(fp_atom_pairs, ATOM_PAIR_NUM_BITS)

        # 2) MACCS (bits)
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        fp_maccs = _bitvect_to_numpy(fp_maccs, MACCS_NUM_BITS)

        # 3) Morgan bit fingerprint
        fp_morgan_bits = GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_NUM_BITS
        )
        fp_morgan_bits = _bitvect_to_numpy(fp_morgan_bits, MORGAN_NUM_BITS)

        # 4) Morgan hashed counts
        fp_morgan_counts = AllChem.GetHashedMorganFingerprint(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_NUM_BITS
        )
        fp_morgan_counts = _sparse_fp_to_numpy(fp_morgan_counts, MORGAN_NUM_BITS)

    except Exception:
        fp_atom_pairs = np.zeros((ATOM_PAIR_NUM_BITS,), dtype=np.float32)
        fp_maccs = np.zeros((MACCS_NUM_BITS,), dtype=np.float32)
        fp_morgan_bits = np.zeros((MORGAN_NUM_BITS,), dtype=np.float32)
        fp_morgan_counts = np.zeros((MORGAN_NUM_BITS,), dtype=np.float32)

    return np.concatenate(
        [fp_atom_pairs, fp_maccs, fp_morgan_bits, fp_morgan_counts], axis=0
    )


class FusionDataset(Dataset):
    """多模态融合任务数据集。

    每个样本包含：
    - graph: PyG Data 图对象
    - desc: 标准化后的描述符特征
    - fp: 分子指纹特征
    - y: 分类标签（LongTensor, shape=[1]）
    """

    def __init__(
        self,
        df,
        featurizer: MoleculeFeaturizer,
        desc_scaler: StandardScaler | None = None,
        fit_desc_scaler: bool = False,
    ):
        self.data = []
        self.desc_scaler = desc_scaler if desc_scaler is not None else StandardScaler()
        print("正在构建多模态数据集...")

        all_descs = []
        temp_data = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            smiles_text = str(row["smiles"])
            mol = Chem.MolFromSmiles(smiles_text)
            if mol is None:
                continue

            # 1) 图结构特征
            graph = featurizer(mol)

            # 2) 分子描述符
            desc = get_descriptors(mol)

            # 3) 指纹特征
            fp = get_fingerprint(mol)

            label = float(row["y"])

            all_descs.append(desc)
            temp_data.append(
                {
                    "graph": graph,
                    "fp": torch.tensor(fp, dtype=torch.float32),
                    "y": torch.tensor([label], dtype=torch.float32),
                }
            )

        if not temp_data:
            raise ValueError("FusionDataset 构建失败：未解析到有效分子，请检查输入数据。")

        # 描述符标准化：仅在训练集 fit，验证/测试集 transform
        if fit_desc_scaler:
            all_descs_scaled = self.desc_scaler.fit_transform(all_descs)
        else:
            all_descs_scaled = self.desc_scaler.transform(all_descs)

        for i, item in enumerate(temp_data):
            item["desc"] = torch.tensor(all_descs_scaled[i], dtype=torch.float32)
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
