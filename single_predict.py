import torch
from dataset.PygMD17 import MD17
from dataset.MolData import MOlDataset
from models import SphereNet
from models.DecNet import DecNet
from scripts.test import run
from rdkit import Chem
import argparse
import os
from e3nn import o3
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='SpChargeNet')
parser.add_argument('--data_root', type=str, default='/home/qunsu/model/SpChargeNet/submit', metavar='N', required=False,
                    help='data_root(npz_dir)')
parser.add_argument('--data_dir', type=str, default='/home/qunsu/model/SpChargeNet/results/delamin_results/ckpt', metavar='N',
                    required=False, help='store path.')
parser.add_argument('--charge_type', type=str, default='large', metavar='N', required=True, help='large, medium, small')
parser.add_argument('--log_dir', type=str, default='', metavar='N', required=False, help='store path.')
parser.add_argument('--file_name', type=str, default='flarge_err.sdf', metavar='N', required=False, help='store path.')
parser.add_argument('--out_name', type=str, default='plarge_err.sdf', metavar='N', required=False, help='store path.')

parser.add_argument('--model_name', type=str, default='SpChargeNet', metavar='N', help='Model name.')
parser.add_argument('--train_size', type=float, default=0.0, metavar='N', help='')
parser.add_argument('--valid_size', type=float, default=0.0, metavar='N', help='')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='')
parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='')
parser.add_argument('--vt_batch_size', type=int, default=1, metavar='N', help='')
parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, metavar='N', help='')
parser.add_argument('--lr_decay_step_size', type=int, default=200, metavar='N', help='')

parser.add_argument('--cutoff', type=float, default=5.0, metavar='N', help='')
parser.add_argument('--num_layers', type=int, default=4, metavar='N', help='')
parser.add_argument('--hidden_channels', type=int, default=128, metavar='N', help='')
parser.add_argument('--out_channels', type=int, default=1, metavar='N', help='')
parser.add_argument('--int_emb_size', type=int, default=64, metavar='N', help='')
parser.add_argument('--basis_emb_size_dist', type=int, default=8, metavar='N', help='')
parser.add_argument('--basis_emb_size_angle', type=int, default=8, metavar='N', help='')
parser.add_argument('--basis_emb_size_torsion', type=int, default=8, metavar='N', help='')
parser.add_argument('--out_emb_channels', type=int, default=256, metavar='N', help='')
parser.add_argument('--num_spherical', type=int, default=3, metavar='N', help='')
parser.add_argument('--num_radial', type=int, default=6, metavar='N', help='')
parser.add_argument('--envelope_exponent', type=int, default=5, metavar='N', help='')
parser.add_argument('--num_before_skip', type=int, default=1, metavar='N', help='')
parser.add_argument('--num_after_skip', type=int, default=2, metavar='N', help='')
parser.add_argument('--num_output_layers', type=int, default=3, metavar='N', help='')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

# npz_path = os.path.join(args.root_dir, args.npz_path)
model_path = os.path.join(args.data_dir, args.charge_type, 'best_checkpoint.pt')
file_path = os.path.join(args.data_root, args.file_name)

test_dataset = MOlDataset(sdf_file=file_path)

charge_model = DecNet(order=1,  # 1 maximum order of spherical harmonics features
                      basis_functions='exp-bernstein',  # exp-bernstein
                      num_basis_functions=128,
                      cutoff=5.0,
                      num_elements=38,
                      radial_MLP=None,
                      avg_num_neighbors=3,
                      correlation=3,
                      num_interactions=3,
                      heads=["dft"],
                      hidden_irreps=o3.Irreps("128x0e+128x1o"),
                      MLP_irreps=o3.Irreps("64x0e"),
                      gate=F.silu,
                      r_max=5.0,
                      num_bessel=8,
                      num_polynomial_cutoff=6)

charge_model = charge_model.to(float)

state = torch.load(model_path, map_location=device)
# print(state)
charge_model.load_state_dict(state['model_state_dict'], strict=True)
    
run_charge = run()
eval_df = run_charge.run(device=device, train_dataset=None, valid_dataset=None,
                         test_dataset=test_dataset,
                         model=charge_model, epochs=args.epochs, batch_size=args.batch_size,
                         vt_batch_size=args.vt_batch_size, lr=args.lr, lr_decay_factor=args.lr_decay_factor,
                         lr_decay_step_size=args.lr_decay_step_size)

target_path = os.path.join(args.data_root, args.out_name)

new_mol = True
save_csv = False

if save_csv:

    eval_df.to_csv(csv_path)

if new_mol:

    writer = Chem.SDWriter(target_path)

    # 遍历分子列表
    for mol, charge in tqdm(zip(eval_df[0], eval_df[1])):

        num_atoms = mol.GetNumAtoms()

        # 更新原子属性并四舍五入
        for i in range(min(num_atoms, len(charge))):
            rounded_value = np.round(charge[i][0], 5)
            mol.GetAtomWithIdx(i).SetProp('molFileAlias', str(rounded_value))

        writer.write(mol)

    writer.close()


