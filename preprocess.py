import os
import glob
import re
import pickle
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2
from multiprocess import yield_filtered_ord_data

# (Assuming the helper functions analyse_rxn, get_matched_fragments, 
# map_atoms, get_changed_atoms, etc. are imported or defined here)
from reaction_center import analyse_rxn, get_matched_fragments

RDLogger.DisableLog('rdApp.*')
# Multiprocessing Worker Setup for Negative Mining
global_rxns =[]
from multiprocess import yield_filtered_ord_data, init_worker, process_single_rule
if __name__ == '__main__':
    # --- PHASE 1: Extract Positives ---
    pb_filepaths = glob.glob(os.path.join("data", "*.pb.gz"))
    # print(pb_filepaths)
    rule_set_positives = defaultdict(list)
    rule_positive_1hop = defaultdict(set)
    rxns =[]
    
    i = 0
    for mapped_smiles in yield_filtered_ord_data(pb_filepaths):
        try:
            result = analyse_rxn(mapped_smiles)
            rxns.append(result.reaction_smiles)
            for j, comp in enumerate(result.components):
                rule_positive_1hop[(result.reaction_smarts, j)].add(comp.neighborhood_smiles)
                rule_set_positives[(result.reaction_smarts, j)].append((i, comp.atom_map_numbers))
            i += 1
        except Exception:
            continue
            
    # Filter for rules with >= 5 positives
    rule_set_positives = {k: v for k, v in rule_set_positives.items() if len(v) >= 5}

    # --- PHASE 2: Mine Hard Negatives (Multiprocessed) ---
    rule_set = defaultdict(set)
    rule_to_frag_indices = defaultdict(list)
    for (rule, frag_idx), pos_list in rule_set_positives.items():
        rule_to_frag_indices[rule].append(frag_idx)
        for rxn_idx, map_nums in pos_list:
            rule_set[rule].add(rxn_idx)

    tasks =[]
    for rule, positives in rule_set.items():
        valid_frags = rule_to_frag_indices[rule]
        hop1_sets = {f: rule_positive_1hop[(rule, f)] for f in valid_frags if (rule, f) in rule_set_positives}
        tasks.append((rule, positives, valid_frags, hop1_sets, 50))

    negatives_dict = defaultdict(list)
    num_cores = 6

    with mp.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(rxns,),
        maxtasksperchild=50       # recycle workers to avoid memory leak
    ) as pool:
        for local_neg_result in tqdm(
            pool.imap_unordered(process_single_rule, tasks),
            total=len(tasks)
        ):
            for key, neg_list in local_neg_result.items():
                negatives_dict[key].extend(neg_list)
    

    # Filter rules lacking at least 5 negatives
    print(len(negatives_dict))
    print(len(rule_set_positives))
    # negatives_dict = {k: v for k, v in negatives_dict.items() if len(v) >= 5}
    # rule_set_positives = {k: v for k, v in rule_set_positives.items() if k in negatives_dict}

    # --- PHASE 3: Save Extracted Data ---
    os.makedirs("extracted_rules_data", exist_ok=True)
    with open("extracted_rules_data/rule_set_positives.pkl", "wb") as f: pickle.dump(rule_set_positives, f)
    with open("extracted_rules_data/negatives_dict.pkl", "wb") as f: pickle.dump(negatives_dict, f)
    with open("extracted_rules_data/rxns.pkl", "wb") as f: pickle.dump(rxns, f)