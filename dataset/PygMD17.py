import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader


class MD17(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`MD17` dataset 
        which is from `"Machine learning of accurate energy-conserving molecular force fields" <https://advances.sciencemag.org/content/3/5/e1603015.short>`_ paper. 
        MD17 is a collection of eight molecular dynamics simulations for small organic molecules. 
    
        Args:
            root (string): The dataset folder will be located at root/name.
            name (string): The name of dataset. Available dataset names are as follows: :obj:`aspirin`, :obj:`benzene_old`, :obj:`ethanol`, :obj:`malonaldehyde`, 
                :obj:`naphthalene`, :obj:`salicylic`, :obj:`toluene`, :obj:`uracil`. (default: :obj:`benzene_old`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        >>> dataset = MD17(name='aspirin')
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(batch=[672], force=[672, 3], pos=[672, 3], ptr=[33], y=[32], z=[672])

        Where the attributes of the output data indicates:
    
        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The property (energy) for the graph (molecule).
        * :obj:`force`: The 3D force for atoms.
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs

    """
    def __init__(self, root='', name='', transform = None, pre_transform = None, pre_filter = None):

        self.name = name
        self.root = root

        super(MD17, self).__init__(self.root, transform, pre_transform, pre_filter)

        print(self.processed_paths[0])

        self.data, self.slices, self.data_size = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.name + '.npz'

    @property
    def processed_file_names(self):
        return self.name + '_pyg.pt'

    def process(self):

        data = np.load(self.raw_paths[0])
        
        # data = np.load('/home/suqun/ChargeData/ChargeData/e4/data.npz')
        # data = np.load('/home/suqun/ChargeData/ChargeData/e78/data.npz')
        # data = np.load('/home/suqun/ChargeData/ChargeData/resp/all/data.npz')

        mol_atom_prop = data['mol_atom_prop']
        positions = data['positions']
        ddec_charge = data['ddec_charges']
        num = data['num_atoms']
        filename = data['filename']

        data_size = len(num)

        data_list = []
        for i in tqdm(range(len(positions))):
            num_i = torch.tensor(num[i], dtype=int)
            prop_i = torch.tensor(mol_atom_prop[i][:num_i], dtype=torch.float32)
            positions_i = torch.tensor(positions[i][:num_i])
            filename_i = filename[i]
            ddec_charge_i = torch.tensor(ddec_charge[i][:num_i])
            data = Data(filename=filename_i, y=ddec_charge_i, pos=positions_i, x=prop_i)

            data_list.append(data)

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices, data_size), self.processed_paths[0])

    def get_idx_split(self, train_size, valid_size, seed):
        ids = shuffle(range(self.data_size), random_state=seed)
        train_size = int(self.data_size * train_size)
        valid_size = int(self.data_size * valid_size)
        # test_size = int(self.data_size - valid_size - train_size)
        train_idx, val_idx, test_idx = ids[:train_size], ids[train_size:train_size + valid_size], ids[train_size + valid_size:]
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

