import torch
import torch.nn.functional as F


class SupConLoss(torch.nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Reference: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        # Ensure features are L2 normalized
        features = F.normalize(features, p=2, dim=1)

        # Compute dot product similarity matrix [Batch_Size, Batch_Size]
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Create a boolean mask: True where labels are identical
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # We don't want an embedding to contrast with itself, so we zero out the diagonal
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        # For numerical stability (prevent exponential overflow)
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        # We divide by the number of positives each item has in the batch
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # The loss is the negative mean
        loss = -mean_log_prob_pos.mean()

        return loss