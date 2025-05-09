import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_features, projection_dim=128):
        super(SimCLRProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, projection_dim)
        )

    def forward(self, x):
        return self.projection(x)

class SimCLR(nn.Module):
    def __init__(self, base_encoder='resnet50', projection_dim=128, pretrained=True):
        super(SimCLR, self).__init__()
        
        # Load the base encoder (ResNet50) pre-trained on ImageNet
        if base_encoder == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            self.encoder_dim = self.encoder.fc.in_features
            
        self.encoder.fc = nn.Identity()  # Remove the final FC layer
        
        # Projection head
        self.projection_head = SimCLRProjectionHead(self.encoder_dim, projection_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, F.normalize(projections, dim=1)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class SimCLRClassifier(nn.Module):
    def __init__(self, base_encoder='resnet50', num_classes=5, projection_dim=128, pretrained=True):
        super(SimCLRClassifier, self).__init__()
        
        # Initialize SimCLR model
        self.simclr = SimCLR(base_encoder, projection_dim, pretrained)
        
        # Classification head
        self.classification_head = ClassificationHead(self.simclr.encoder_dim, num_classes)
        
    def forward(self, x):
        features, _ = self.simclr(x)
        logits = self.classification_head(features)
        return logits
    
    def get_features(self, x):
        features, _ = self.simclr(x)
        return features 