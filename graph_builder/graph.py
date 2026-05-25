from rdkit import Chem
from .featurizer import get_atom_features, get_bond_features
import torch
from torch_geometric.data import Data


def build_reaction_graph(mapped_rxn_smiles, target_map_nums):
    """
    Converts the reactant portion of a mapped SMILES into a PyG Data object.
    Creates a 1-hop mask based on the target_map_nums.
    """
    # 1. Extract Reactant
    reactants_smiles = mapped_rxn_smiles.split(">")[0]
    mol = Chem.MolFromSmiles(reactants_smiles)
    
    if mol is None:
        return None
    
    num_nodes = mol.GetNumAtoms()
    
    # 2. Construct Node Features (x)
    node_features =[]
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 3. Construct Edge Index and Edge Features (edge_attr)
    edge_indices = []
    edge_features =[]
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_feat = get_bond_features(bond)
        
        # PyG uses directed edges, so we must add both i->j and j->i for undirected graphs
        edge_indices += [[i, j], [j, i]]
        edge_features += [bond_feat, bond_feat]
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Handle molecules with no bonds (e.g., isolated ions)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(get_bond_features(Chem.MolFromSmiles("CC").GetBonds()[0]))), dtype=torch.float)
        
    # 4. Generate the 1-hop Mask
    target_map_set = set(target_map_nums)
    center_indices =[]
    hop1_indices = set()
    
    # Find the RDKit indices of the target atoms using their Atom Map Numbers
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in target_map_set:
            idx = atom.GetIdx()
            center_indices.append(idx)
            hop1_indices.add(idx)
            # Add neighbors (1-hop)
            for neighbor in atom.GetNeighbors():
                hop1_indices.add(neighbor.GetIdx())
                
    # Create a boolean mask of shape[num_nodes]
    # True if the atom is in the reaction center OR its immediate 1-hop neighbor
    hop1_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if hop1_indices:
        hop1_mask[list(hop1_indices)] = True
        
    # Optional: Keep a center-only mask just in case you want to ablate/compare later
    center_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if center_indices:
        center_mask[center_indices] = True

    # 5. Build PyG Data Object
    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        hop1_mask=hop1_mask,
        center_mask=center_mask
    )
    
    return data