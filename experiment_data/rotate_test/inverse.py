from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np
import copy

# 加载分子（确保含有3D构象）
mol = Chem.MolFromMolFile("example.sdf", removeHs=False)
conf = mol.GetConformer()

# 准备输出分子集合
rotated_mols = []

for i in range(100):
    new_mol = copy.deepcopy(mol)
    conf = new_mol.GetConformer()

    # 平移：生成一个随机向量（[-5, 5] Å范围）
    translation = np.random.uniform(-5, 5, size=3)
    for atom_id in range(new_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_id)
        new_pos = pos + translation
        conf.SetAtomPosition(atom_id, new_pos)

    # 旋转：生成随机旋转矩阵
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    psi = np.random.uniform(0, 2*np.pi)

    def rotation_matrix_z(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])

    def rotation_matrix_y(angle):
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

    def rotation_matrix_x(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

    R = rotation_matrix_z(psi) @ rotation_matrix_y(phi) @ rotation_matrix_x(theta)

    # 获取中心点
    centroid = rdMolTransforms.ComputeCentroid(conf)
    for atom_id in range(new_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_id)
        v = np.array([pos.x - centroid.x, pos.y - centroid.y, pos.z - centroid.z])
        v_rot = R @ v
        new_pos = v_rot + np.array([centroid.x, centroid.y, centroid.z])
        conf.SetAtomPosition(atom_id, new_pos)

    rotated_mols.append(new_mol)

# 写入 SDF 文件
w = Chem.SDWriter("rotated_conformers.sdf")
for m in rotated_mols:
    w.write(m)
w.close()

