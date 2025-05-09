import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5, world_size=1):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        
        # Create a mask for matching positive pairs
        self.mask = self._get_correlated_mask().cuda()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self):
        diag = torch.eye(2 * self.batch_size)
        l1 = torch.diag(torch.ones(self.batch_size), diagonal=self.batch_size)
        l2 = torch.diag(torch.ones(self.batch_size), diagonal=-self.batch_size)
        mask = (1 - diag - l1 - l2).bool()
        return mask

    def forward(self, z_i, z_j):
        """
        Calculate the NT-Xent loss for SimCLR
        
        Args:
            z_i: first projection batch of size [batch_size, projection_dim]
            z_j: second projection batch of size [batch_size, projection_dim]
        """
        # Get batch size
        batch_size = z_i.shape[0]
        
        # Update mask if batch size changed
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.mask = self._get_correlated_mask().cuda()
        
        # Stack representations
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Filter out the scores from the positive pairs
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        
        # Get negative samples mask
        negatives = similarity_matrix[self.mask].view(2 * self.batch_size, -1)
        
        # Compute logits
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        
        # Create labels for cross-entropy (first is positive pair)
        labels = torch.zeros(2 * self.batch_size).cuda().long()
        
        # Compute cross entropy
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        
        return loss 