import os
import glob
import re
import pickle
import multiprocessing as mp
from collections import defaultdict

from tqdm import tqdm
from attr import dataclass
from rdkit import Chem
from rdkit import RDLogger
import logging

from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2

RDLogger.DisableLog('rdApp.*') #type: ignore

@dataclass
class ComponentResult:
    """Analysis result for a single connected component of changed atoms."""
    atom_map_numbers: list[int]   # atom indices in the merged product mol
    smarts_reactant: str          # SMARTS of this component on the reactant side
    neighborhood_smiles: str

@dataclass
class ReactionCenterResult:
    """Full result of reaction-center analysis."""
    components: list[ComponentResult]
    reaction_smarts: str
    reaction_smiles: str

def map_atoms(smiles):
    reactants, _, products = smiles.split(">")
    reactant_mol = Chem.MolFromSmiles(reactants)
    product_mol = Chem.MolFromSmiles(products)
    
    max_map_num = 0
    # 1. Check max map number in BOTH reactant and product
    for atom in reactant_mol.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            max_map_num = max(max_map_num, atom.GetAtomMapNum())
    for atom in product_mol.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            max_map_num = max(max_map_num, atom.GetAtomMapNum())
            
    # 2. Assign unique map numbers to unmapped atoms in reactant
    for atom in reactant_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            max_map_num += 1
            atom.SetAtomMapNum(max_map_num)
            
    # 3. Assign unique map numbers to unmapped atoms in product (entering groups)
    for atom in product_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            max_map_num += 1
            atom.SetAtomMapNum(max_map_num)
            
    return Chem.MolToSmiles(reactant_mol, isomericSmiles=False, canonical=True) + ">>" + Chem.MolToSmiles(product_mol, isomericSmiles=False, canonical=True)

def get_changed_atoms(reactant_mol, product_mol):
    changed_atoms = set()
    r_atoms = {r.GetAtomMapNum(): r for r in reactant_mol.GetAtoms()}
    p_atoms = {p.GetAtomMapNum(): p for p in product_mol.GetAtoms()}
    common_atoms = set(r_atoms.keys()).intersection(set(p_atoms.keys()))
    
    for map_no in common_atoms:
        r_atom = r_atoms[map_no]
        p_atom = p_atoms[map_no]
        if r_atom.GetSymbol() != p_atom.GetSymbol():
            raise ValueError(f"Atom map {map_no} has different symbols in reactant and product")
        if (r_atom.GetDegree() != p_atom.GetDegree() or 
            r_atom.GetFormalCharge() != p_atom.GetFormalCharge() or 
            r_atom.GetHybridization() != p_atom.GetHybridization()):
            # print(f"Atom map {map_no} has different properties in reactant and product")
            changed_atoms.add(map_no)
            
    bonds_reactant = set()
    for bond in reactant_mol.GetBonds():
        at1, at2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if at1 > at2: at1, at2 = at2, at1
        bonds_reactant.add((at1, at2, bond.GetBondType()))
    
    bonds_product = set()
    for bond in product_mol.GetBonds():
        at1, at2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if at1 > at2: at1, at2 = at2, at1
        bonds_product.add((at1, at2, bond.GetBondType()))

    bonds_changed = bonds_reactant.symmetric_difference(bonds_product)
    for at1, at2, _ in bonds_changed:
        if at1 in common_atoms or at2 in common_atoms:
            changed_atoms.update([at1, at2])
            
    return changed_atoms

def get_components(indices, r_mol):
    adj_list = defaultdict(set)
    for bond in r_mol.GetBonds():
        at1, at2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if at1 in indices and at2 in indices:
            adj_list[at1].add(at2)
            adj_list[at2].add(at1)
    visited = set()
    components =[]
    for idx in indices:
        if idx not in visited:
            stack = [idx]
            comp =[]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    stack.extend(adj_list[node] - visited)
            components.append(comp)
    return components

def get_components_pmol(indices, pmol):
    adj_list = defaultdict(set)
    indices_ = {a.GetAtomMapNum() for a in pmol.GetAtoms() if a.GetAtomMapNum() in indices}
    for bond in pmol.GetBonds():
        at1, at2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if at1 in indices_ and at2 in indices_:
            adj_list[at1].add(at2)
            adj_list[at2].add(at1)
    visited = set()
    components =[]
    for idx in indices_:
        if idx not in visited:
            stack = [idx]
            comp =[]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    stack.extend(adj_list[node] - visited)
            components.append(comp)
    return components

def get_1_hop(mol, indices):
    new_indices_list = set()
    for i in indices:
        new_indices_list.add(i)
        atom = mol.GetAtomWithIdx(i)
        for nb in atom.GetNeighbors():
            new_indices_list.add(nb.GetIdx())
    return new_indices_list

def extract_smarts_by_map_nums(mol, target_map_nums):
    target_map_set = set(target_map_nums)
    internal_indices =[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() in target_map_set]
            
    if not internal_indices:
        # print("Warning: No atoms found with the provided map numbers.",)
        # print(Chem.MolToSmiles(mol))
        # print(target_map_nums)
        raise ValueError(f"No atoms found with the provided map numbers")
        return "", ""

    indices_extended = get_1_hop(mol, internal_indices)
    fragment_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=indices_extended)
    rwmol = Chem.RWMol(mol)
    
    hyb_map = {
        Chem.HybridizationType.SP: "^1",
        Chem.HybridizationType.SP2: "^2",
        Chem.HybridizationType.SP3: "^3"
    }
    for idx in internal_indices:
        atom = rwmol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol().lower() if atom.GetIsAromatic() else atom.GetSymbol()
        charge = atom.GetFormalCharge()
        charge_str = f"+{charge}" if charge > 0 else f"{charge}" if charge < 0 else ""
        custom_atom_smarts = f"[{symbol}{charge_str}]"
        query_atom = Chem.AtomFromSmarts(custom_atom_smarts)
        rwmol.ReplaceAtom(idx, query_atom)
        
    return Chem.MolFragmentToSmarts(rwmol, atomsToUse=internal_indices), fragment_smiles

def clear_atom_map_numbers(smiles_with_maps):
    mol = Chem.MolFromSmiles(smiles_with_maps, sanitize=False)
    if mol is None: return ""
    mol.UpdatePropertyCache(strict=False)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)

def analyse_rxn(smiles):
    mapped_smiles = map_atoms(smiles)
    reactants, _, products = mapped_smiles.split(">")
    r_mol, p_mol = Chem.MolFromSmiles(reactants), Chem.MolFromSmiles(products)
    
    changed_atoms = get_changed_atoms(r_mol, p_mol)
    # print(f"Changed atom map numbers: {changed_atoms}")
    components = get_components(changed_atoms, r_mol)
    components_p = get_components_pmol(changed_atoms, p_mol)
    
    # Pack extracted SMARTS, neighborhood SMILES, and the component map numbers together to avoid sorting desync
    smarts_templates_rmol =[]
    for comp in components:
        smarts, nbr = extract_smarts_by_map_nums(r_mol, comp)
        smarts_templates_rmol.append((smarts, nbr, comp))
        
    smarts_templates_pmol =[]
    for comp in components_p:
        smarts, nbr = extract_smarts_by_map_nums(p_mol, comp)
        smarts_templates_pmol.append((smarts, nbr, comp))
        
    # Sort them using SMARTS and neighborhood SMILES as sort keys
    smarts_templates_rmol.sort(key=lambda x: (x[0], x[1]))
    smarts_templates_pmol.sort(key=lambda x: (x[0], x[1]))
    
    c_list =[]
    for smarts, nbr, comp in smarts_templates_rmol:
        c = ComponentResult(
            atom_map_numbers=comp,
            smarts_reactant=smarts,
            neighborhood_smiles=clear_atom_map_numbers(nbr)
        )
        c_list.append(c)
        
    smarts_r = [item[0] for item in smarts_templates_rmol]
    smarts_p = [item[0] for item in smarts_templates_pmol]
    reaction_smarts = '.'.join(smarts_r) + '>>' + '.'.join(smarts_p)
    
    return ReactionCenterResult(components=c_list, reaction_smarts=reaction_smarts, reaction_smiles=mapped_smiles)

def split_by_lengths(data, lengths):
    it = iter(data)
    return [[next(it) for _ in range(length)] for length in lengths]

def get_matched_fragments(rmol, template_mol):
    """Extracts 1-hop fragments and their positional index for a given rule using pre-parsed Mols."""
    if rmol is None or template_mol is None: 
        return []
        
    match = rmol.GetSubstructMatch(template_mol)
    if not match: 
        return []
        
    template_frags = Chem.GetMolFrags(template_mol)
    idx_to_cut = [len(frag) for frag in template_frags]
    match_split = split_by_lengths(match, idx_to_cut)
    
    ans = []
    for frag_idx, m in enumerate(match_split):
        match_extended = get_1_hop(rmol, m) 
        fragment_smiles = Chem.MolFragmentToSmiles(rmol, atomsToUse=match_extended)
        map_n = [rmol.GetAtomWithIdx(i).GetAtomMapNum() for i in m]
        ans.append((frag_idx, map_n, clear_atom_map_numbers(fragment_smiles)))
        
    return ans