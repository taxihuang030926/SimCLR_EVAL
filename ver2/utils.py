import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def visualize_embeddings(model, data_loader, device, num_samples=1000):
    """
    Visualize embeddings using t-SNE
    """
    # Get features and labels
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            if len(features) * images.size(0) >= num_samples:
                break
                
            images = images.to(device)
            batch_features = model.get_features(images).cpu().numpy()
            
            features.append(batch_features)
            labels.append(targets.numpy())
    
    # Concatenate
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    # Limit number of samples
    if features.shape[0] > num_samples:
        indices = np.random.choice(features.shape[0], num_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    category_names = ['Animals', 'Vehicles', 'Plants', 'Buildings and Structures', 'Clothing and Accessories']
    
    for i in range(5):  # 5 categories
        indices = labels == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=category_names[i], alpha=0.6)
    
    plt.legend()
    plt.title('t-SNE visualization of feature embeddings')
    plt.savefig('tsne_embeddings.png')
    plt.close()

def evaluate_model(model, data_loader, device):
    """
    Evaluate model and return detailed metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Generate classification report
    category_names = ['Animals', 'Vehicles', 'Plants', 'Buildings and Structures', 'Clothing and Accessories']
    report = classification_report(all_targets, all_preds, target_names=category_names)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(category_names))
    plt.xticks(tick_marks, category_names, rotation=45)
    plt.yticks(tick_marks, category_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return report 