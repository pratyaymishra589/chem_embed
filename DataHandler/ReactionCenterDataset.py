import torch
from torch.utils.data import Dataset

from tqdm import tqdm

class ReactionCenterDataset(Dataset):
    def __init__(self, rule_set_positives, negatives_dict, rxns, build_graph_fn, device):
        self.data =[]
        self.rule_to_id = {}
        self.device = device
        current_id = 0

        # 1. Add Positives
        for rule_key, pos_list in rule_set_positives.items():
            if rule_key not in self.rule_to_id:
                self.rule_to_id[rule_key] = current_id
                current_id += 1

            rule_id = self.rule_to_id[rule_key]

            for rxn_idx, map_nums in pos_list:
                self.data.append({
                    'rxn_smiles': rxns[rxn_idx],
                    'map_nums': map_nums,
                    'rule_id': rule_id,
                    'is_positive': True
                })

        # 2. Add Hard Negatives
        for rule_key, neg_list in negatives_dict.items():
            if rule_key in self.rule_to_id:
                rule_id = self.rule_to_id[rule_key]

                for rxn_idx, map_nums in neg_list:
                    self.data.append({
                        'rxn_smiles': rxns[rxn_idx],
                        'map_nums': map_nums,
                        'rule_id': rule_id,
                        'is_positive': False
                    })

        # 3. Precompute all graphs into memory so training is lightning fast
        self.precomputed_graphs =[]
        offset = len(self.rule_to_id)

        print(f"Precomputing {len(self.data)} PyG graphs into RAM...")
        for item in tqdm(self.data, desc="Building Graphs"):
            # Build the graph once
            graph = build_graph_fn(item['rxn_smiles'], item['map_nums'])

            # Attach the contrastive labels
            contrastive_label = item['rule_id'] if item['is_positive'] else item['rule_id'] + offset
            graph.y = torch.tensor([contrastive_label], dtype=torch.long)
            graph.rule_id = item['rule_id']
            graph.is_positive = item['is_positive']
            graph = graph.to(self.device)
            self.precomputed_graphs.append(graph)

    def __len__(self):
        return len(self.precomputed_graphs)

    def __getitem__(self, idx):
        # Instantly return the pre-built graph from memory
        return self.precomputed_graphs[idx]