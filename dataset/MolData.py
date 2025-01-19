import torch
from torch_geometric.data import Batch, Data, Dataset
from rdkit import Chem
import numpy as np


class MOlDataset(Dataset):
    def __init__(self, sdf_file):
        """
        初始化数据集，读取 SDF 文件并提取每个分子。

        :param sdf_file: 包含多个分子的 SDF 文件路径
        :param transform: 可选的转换操作（如图像预处理、数值归一化等）
        """
        # 读取 SDF 文件并解析
        self.sdf_file = sdf_file

        # 使用 RDKit 读取 SDF 文件中的所有分子
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        self.molecules = [mol for mol in suppl if mol is not None]  # 过滤掉无效的分子

    def __len__(self):
        """
        返回数据集的大小（分子数量）
        """
        return len(self.molecules)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本（分子）。

        :param idx: 样本的索引
        :return: 处理后的数据样本（例如输入特征）
        """
        # 获取分子
        mol = self.molecules[idx]

        # 提取特征（例如使用分子指纹或其他方式）
        features = self.extract_features(mol)
        atom_position = torch.tensor(mol.GetConformers()[0].GetPositions())

        mol_data = Data(mol=mol, x=torch.tensor(features, dtype=torch.float32), pos=torch.tensor(atom_position, dtype=torch.float32))

        return mol_data

    def extract_features(self, mol):
        """
        提取分子的特征，示例中是使用 Morgan Fingerprints（类似于指纹）来表示分子。

        :param mol: RDKit 分子对象
        :return: 分子的特征向量（通常是固定长度的向量）
        """
        # 生成 Morgan Fingerprint（常见的分子指纹）
        mol_atom_prop = []

        for atom in mol.GetAtoms():
            atom_prop = atom_features(atom)
            mol_atom_prop.append(atom_prop)

        return mol_atom_prop


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H',
                                           'Unknown']) +  # H?
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()] +
                    [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
                    one_of_k_encoding_unk(atom.GetHybridization(), [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                          SP3D, Chem.rdchem.HybridizationType.SP3D2]))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



