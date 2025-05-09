import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, feature_dim=128, num_classes=5):
        super(SimCLR, self).__init__()
        
        # Load pre-trained ResNet
        self.encoder = models.resnet50(pretrained=True)
        self.feature_dim = feature_dim
        
        # Remove the final fully connected layer
        self.encoder.fc = nn.Identity()
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        proj = self.projector(features)
        proj = nn.functional.normalize(proj, dim=1)
        
        # Classification output
        logits = self.classifier(features)
        
        return proj, logits

class NT_Xent(nn.Module):
    def __init__(self, temperature=0.5):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        N = 2 * batch_size
        
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.mm(z, z.t().contiguous()) / self.temperature
        
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = torch.ones((N, N), dtype=bool).fill_diagonal_(0)
        
        negative_samples = sim[mask].reshape(N, -1)
        
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss 