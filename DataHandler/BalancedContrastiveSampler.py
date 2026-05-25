import random
import torch
from torch.utils.data.sampler import Sampler


class BalancedContrastiveSampler(Sampler):
    """
    Ensures that every batch contains:
    - K different rules (providing Soft Negatives)
    - P positives for each of those K rules
    - N hard negatives for each of those K rules
    """
    def __init__(self, dataset, rules_per_batch=4, pos_per_rule=4, neg_per_rule=4):
        self.dataset = dataset
        self.rules_per_batch = rules_per_batch
        self.pos_per_rule = pos_per_rule
        self.neg_per_rule = neg_per_rule

        self.batch_size = rules_per_batch * (pos_per_rule + neg_per_rule)

        # Group indices by (rule_id, is_positive)
        self.idx_map = {}
        for idx, item in enumerate(dataset.data):
            key = (item['rule_id'], item['is_positive'])
            if key not in self.idx_map:
                self.idx_map[key] = []
            self.idx_map[key].append(idx)

        # Get list of unique rule IDs that have BOTH positives and negatives
        self.valid_rules =[]
        for rule_id in range(len(dataset.rule_to_id)):
            has_pos = len(self.idx_map.get((rule_id, True),[])) >= pos_per_rule
            has_neg = len(self.idx_map.get((rule_id, False),[])) >= neg_per_rule
            if has_pos and has_neg:
                self.valid_rules.append(rule_id)

    def __iter__(self):
        # Determine how many batches we can form
        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            batch_indices =[]

            # 1. Randomly sample K rules (Soft Negatives)
            sampled_rules = random.sample(self.valid_rules, self.rules_per_batch)

            for rule_id in sampled_rules:
                # 2. Sample P positives
                pos_indices = random.sample(self.idx_map[(rule_id, True)], self.pos_per_rule)
                # 3. Sample N hard negatives
                neg_indices = random.sample(self.idx_map[(rule_id, False)], self.neg_per_rule)

                batch_indices.extend(pos_indices)
                batch_indices.extend(neg_indices)

            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size