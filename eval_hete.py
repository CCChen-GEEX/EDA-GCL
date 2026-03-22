import torch
import functools
import numpy as np
from abc import ABC, abstractmethod
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import f1_score


# =============================================================================
# Data Split Utilities
# =============================================================================

def get_split(data, num):
    """
    Extract train/valid/test splits from data masks.
    
    Args:
        data: Dataset object containing mask tensors
        num: Index of the split to extract
    
    Returns:
        Dictionary containing train, valid, and test indices
    """
    train_mask = data.train_mask[:, num].cpu()
    val_mask = data.val_mask[:, num].cpu()
    test_mask = data.test_mask[:, num].cpu()
    
    num_samples = data.x.shape[0]
    indices = torch.arange(num_samples)

    return {
        'train': indices[train_mask],
        'valid': indices[val_mask],
        'test': indices[test_mask]
    }


# =============================================================================
# Classifier Models
# =============================================================================

class LogisticRegression(nn.Module):
    """Logistic Regression classifier for evaluation."""
    
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


# =============================================================================
# Evaluator Base Classes
# =============================================================================

class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split
        result = self.evaluate(x, y, split)
        return result


class LREvaluator(BaseEvaluator):
    """Logistic Regression Evaluator with training loop."""
    
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        
        # Initialize classifier
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(
            classifier.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        # Tracking best metrics
        best_val_acc = 0
        best_test_acc = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(
            total=self.num_epochs, 
            desc='(LR)',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        ) as pbar:
            for epoch in range(self.num_epochs):
                # ----- Training Step -----
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])
                loss.backward()
                optimizer.step()

                # ----- Evaluation Step -----
                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    
                    # Test metrics
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                    test_acc = (y_test == y_pred).mean()

                    # Validation metrics
                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_acc = (y_val == y_pred).mean()

                    # Update best metrics
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({
                        'best test ACC': best_test_acc, 
                        'F1Mi': best_test_micro, 
                        'F1Ma': best_test_macro
                    })
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'acc': best_test_acc,
        }


# =============================================================================
# Decorator Utilities
# =============================================================================

def repeat(n_times):
    """
    Decorator to repeat evaluation multiple times and compute statistics.
    
    Args:
        n_times: Number of repetitions
    
    Returns:
        Decorated function that returns mean and std statistics
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    """
    Convert probability predictions to one-hot encoding.
    
    Args:
        y_pred: Probability matrix of shape (n_samples, n_classes)
    
    Returns:
        Boolean one-hot matrix
    """
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


# =============================================================================
# Output Utilities
# =============================================================================

def print_statistics(statistics, function_name):
    """Print evaluation statistics in formatted output."""
    print(f'(E) | {function_name}:', end=' ')
    
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


# =============================================================================
# Main Evaluation Function
# =============================================================================

def label_classification_hete(embeddings, y, data, test_repeat=10):
    """
    Evaluate embeddings using logistic regression with multiple random splits.
    
    Args:
        embeddings: Node embeddings from the model
        y: Ground truth labels
        data: Dataset object containing masks
        test_repeat: Number of evaluation repetitions
    
    Returns:
        Tuple of (mean, std) for micro_f1, macro_f1, and accuracy (scaled to 0-100)
    """
    micro_f1 = torch.zeros(test_repeat)
    macro_f1 = torch.zeros(test_repeat)
    acc = torch.zeros(test_repeat)
    
    for num in range(test_repeat):  
        split = get_split(data, num)
        logreg = LREvaluator(num_epochs=20000)
        result = logreg.evaluate(embeddings, y, split)
        
        micro_f1[num] = result['micro_f1']
        macro_f1[num] = result['macro_f1']
        acc[num] = result['acc']
    
    # Print intermediate results
    print('micro_f1:', micro_f1.mean().item(), 'std:', micro_f1.std().item())
    print('macro_f1:', macro_f1.mean().item(), 'std:', macro_f1.std().item())
    print('acc:', acc.mean().item(), 'std:', acc.std().item())
    
    # Return scaled results (0-100)
    return (
        micro_f1.mean().item() * 100, micro_f1.std().item() * 100,
        macro_f1.mean().item() * 100, macro_f1.std().item() * 100,
        acc.mean().item() * 100, acc.std().item() * 100
    )