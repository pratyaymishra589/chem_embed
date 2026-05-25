import re
from collections import defaultdict
from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2
from reaction_center import get_matched_fragments
import random

import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

# Pre-compile regex for speed
import re
from concurrent.futures import ProcessPoolExecutor
from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2
from rdkit import Chem
from rdkit.Chem import Descriptors

# Pre-compile regex for performance
MAP_REGEX = re.compile(r':(\d+)\]')

def has_duplicate_maps(smi_section):
    """Checks for duplicate atom mappings efficiently using an iterator."""
    found = set()
    for match in MAP_REGEX.finditer(smi_section):
        map_id = match.group(1)
        if map_id in found:
            return True
        found.add(map_id)
    return False

def filter_mapped_reaction(rxn_str, mw_min=50, atom_min=4):
    """
    Filters reaction components by weight and atom count.
    Uses UpdatePropertyCache to avoid the RuntimeError with MolWt.
    """
    sections = rxn_str.split('>')
    new_sections = []

    for section in sections:
        if not section:
            new_sections.append("")
            continue
            
        valid_components = []
        for smi in section.split('.'):
            # Use MolFromSmiles with sanitize=False for speed
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if not mol:
                continue
            
            # Fast check: Heavy atoms (doesn't require valence calculation)
            heavy_atoms = mol.GetNumHeavyAtoms()
            if heavy_atoms < atom_min:
                continue
            
            # FIX: Explicitly calculate valence/hydrogens so MolWt doesn't crash
            try:
                mol.UpdatePropertyCache(strict=False)
                mw = Descriptors.MolWt(mol)
                if mw >= mw_min:
                    valid_components.append(smi)
            except Exception:
                # If valence calculation fails, the SMILES is likely too broken to use
                continue
        
        new_sections.append('.'.join(valid_components))

    return '>'.join(new_sections)

def process_single_file(pb_filepath):
    """Processes one protocol buffer file. Used for multiprocessing."""
    results = []
    try:
        dataset = message_helpers.load_message(pb_filepath, dataset_pb2.Dataset)
    except Exception:
        return results

    for rxn in dataset.reactions:
        # Early Exit 1: Fast integer counts
        num_reactants = len(rxn.inputs)
        if num_reactants > 3 or num_reactants == 0: continue
        
        num_products = sum(len(outcome.products) for outcome in rxn.outcomes) if rxn.outcomes else 0
        if num_products > 3 or num_products == 0: continue

        # Find CXSMILES identifier
        cx_smiles_raw = None
        for identifier in rxn.identifiers:
            if identifier.type == identifier.REACTION_CXSMILES and ':' in identifier.value:
                cx_smiles_raw = identifier.value
                break
        
        if not cx_smiles_raw:
            continue

        # Clean CXSMILES and split
        cx_smiles = cx_smiles_raw.split(" ")[0]
        parts = cx_smiles.split(">")
        if len(parts) < 3: continue
        
        reactants, products = parts[0], parts[2]

        # Early Exit 2: Efficient Duplicate Check
        if has_duplicate_maps(reactants) or has_duplicate_maps(products):
            continue

        # Processing & Filtering
        rxn_smarts = f"{reactants}>>{products}"
        filtered_rxn = filter_mapped_reaction(rxn_smarts)
        
        # Verify the reaction still has valid reactants and products after filtering
        if '>>' in filtered_rxn:
            sides = filtered_rxn.split('>>')
            if len(sides) == 2 and all(sides):
                results.append(filtered_rxn)
            
    return results

def yield_filtered_ord_data(file_paths, n_workers=None):
    """Main generator leveraging multiprocessing for CPU-bound tasks."""
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for file_results in executor.map(process_single_file, file_paths):
            for rxn_smarts in file_results:
                yield rxn_smarts

                
# Worker memory initializer
global_rxns =[]
def init_worker(shared_rxns):
    global global_rxns
    global_rxns = shared_rxns

def process_single_rule(args):
    rule, positives, valid_frags, hop1_sets, max_negs = args
    global global_rxns
    local_negatives = defaultdict(list)
    
    # 1. Identify all possible indices that are NOT positives for this rule
    total_rxns = len(global_rxns)
    all_indices = list(range(total_rxns))
    # Filter out known positives to get candidate indices
    candidate_indices = [idx for idx in all_indices if idx not in positives]
    
    # 2. Randomly sample 5000 candidates (or fewer if the dataset is small)
    sample_size = len(candidate_indices)
    sampled_search_space = random.sample(candidate_indices, k=sample_size)
    
    # 3. Search only within the sampled 5000 reactions
    for i in sampled_search_space:
        # Check if we already have enough negatives for ALL fragments in this rule
        if all(len(local_negatives[(rule, f_idx)]) >= max_negs for f_idx in valid_frags):
            break
            
        rxn_smiles = global_rxns[i]
        negatives = get_matched_fragments(rxn_smiles, rule)
        
        for frag_idx, map_nums, frag_smiles in negatives:
            key = (rule, frag_idx)
            # Only process if this fragment was part of our "valid" (>=5 positives) list
            if key in hop1_sets and len(local_negatives[key]) < max_negs:
                # The "Unseen Context" test: Check if this 1-hop environment is a known positive
                if frag_smiles not in hop1_sets[key]:
                    local_negatives[key].append((i, map_nums))
                    
    return local_negatives