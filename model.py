"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import torch
import tempfile
import urllib.request
import gzip
from pathlib import Path

DOWNLOAD_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
DATASET_DIR = Path(tempfile.gettempdir())

def fetch_dataset_file(name):
    path = DATASET_DIR / f'{name}.gz'
    if not path.exists():
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(DOWNLOAD_URL + name + '.gz', tmp)
        tmp.replace(path)
    with gzip.open(path, 'rb') as f:
        return f.read()

def load_dataset_arr(name, n=None):
    is_img = "images" in name
    data = fetch_dataset_file(name)
    
    arr = np.frombuffer(data, np.uint8, offset=16 if is_img else 8)
    if is_img:
        arr = arr.reshape(-1, 28, 28)
    if n is not None:
        arr = arr[:n]

    if is_img:
        return torch.from_numpy(arr.astype(np.float32) / 255.0)
    return torch.from_numpy(arr.astype(np.int64))

def load_fashion_mnist(n_train=10000, n_test=2000):
    # Download the four idx gz files once, parse with np.frombuffer, return float tensors in 0-1 and int64 labels.
    X_train = load_dataset_arr("train-images-idx3-ubyte", n_train)
    y_train = load_dataset_arr("train-labels-idx1-ubyte", n_train)
    X_test = load_dataset_arr("t10k-images-idx3-ubyte", n_test)
    y_test = load_dataset_arr("t10k-labels-idx1-ubyte", n_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

# Step 2 - FashionDataset
class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        # TODO: store X, y, mean, std
        super().__init__()
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        # TODO: ((X[i] - mean) / std, y[i])
        return ((self.X[i] - self.mean) / self.std, self.y[i])

# Step 3 - make_loaders
from torch.utils.data import DataLoader

def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    # TODO: last val_size training images -> validation; seeded shuffled train loader; return loaders + sizes.
    X_train = data["X_train"]
    y_train = data["y_train"]

    n_train = len(X_train) - val_size

    train_ds = FashionDataset(X_train[:n_train], y_train[:n_train])
    val_ds = FashionDataset(X_train[n_train:], y_train[n_train:])
    test_ds = FashionDataset(data["X_test"], data["y_test"])

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed)),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "sizes": (len(train_ds), len(val_ds), len(test_ds))
    }

# Step 4 - MLP
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden1=300, hidden2=100, n_classes=10):
        super().__init__()
        # TODO: fc1 (784 -> hidden1), fc2 (hidden1 -> hidden2), out (hidden2 -> n_classes)
        self.fc1 = nn.Linear(28 * 28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        # TODO: flatten, ReLU after fc1 and fc2, return logits
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def count_parameters(model):
    # TODO: number of trainable parameters as an int
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 5 - train_one_epoch
def train_one_epoch(model, loader, loss_fn, optimizer):
    # TODO: model.train(); per batch: zero_grad, forward, loss, backward, step; return mean batch loss.
    model.train()
    total = 0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

# Step 6 - evaluate
def evaluate(model, loader, loss_fn):
    # TODO: eval mode + no_grad; return (mean loss over examples, accuracy) as floats.
    model.eval()
    total_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total_loss += loss_fn(logits, y).item()  * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n+=len(y)
    return total_loss / n, correct / n

# Step 7 - fit
import copy

def fit(model, loaders, epochs=5, lr=0.05, seed=42):
    # TODO: seeded SGD training with a validation pass per epoch; restore the best-val-accuracy weights; return history.
    torch.manual_seed(seed)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = -1
    best_epoch = -1

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, loaders["val"], loss_fn)

        if val_acc >= best_val_acc:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    
    model.load_state_dict(best_state)

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "best_epoch": best_epoch,
    }

# Step 8 - lr_range_test
def lr_range_test(make_model, loader, lrs, n_batches=20, seed=42):
    # TODO: fresh seeded model per lr; mean loss over the first n_batches; return {lr: loss}.
    result = {}
    for lr in lrs:
        torch.manual_seed(seed)
        
        model = make_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        total_loss = 0
        count = 0
        for data, target in loader:
            count += 1

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if count >= 20:
                result[lr] = total_loss / count
                break

    return result

# Step 9 - random_search
import torch
def random_search(loaders, n_trials=4, epochs=2, seed=42):
    # TODO: seeded random configurations of (hidden1, hidden2, lr); fit each; return trials and the best.
    trials = []
    best_val_acc = -1
    
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        torch.manual_seed(seed)
        hidden1 = rng.choice([100,200,300])
        hidden2 = rng.choice([50,100])
        lr = rng.choice([0.01, 0.05, 0.1])

        model = MLP(hidden1, hidden2)
        history = fit(model, loaders, epochs, lr, seed)

        val_acc = max(history["val_acc"])
        trial = {
            "hidden1": hidden1,
            "hidden2": hidden2,
            "lr": lr,
            "val_acc": val_acc
        }
        trials.append(trial)

        if val_acc > best_val_acc:
            best_trial = trial

    return {
        "trials": trials,
        "best": best_trial
    }

# Step 10 - test_accuracy
def test_accuracy(model, loaders):
    # TODO: accuracy on loaders['test'] via evaluate.
    return evaluate(model, loaders["test"], nn.CrossEntropyLoss())[1]

# Step 11 - save_model
def save_model(model, path):
    # TODO: torch.save({'state_dict': ..., 'config': {'hidden1', 'hidden2', 'n_classes'}}, path)
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'hidden1': model.fc1.out_features,
            'hidden2':model.fc2.out_features,
            'n_classes': model.out.out_features 
        }
    },
    path)

def load_model(path):
    # TODO: rebuild MLP from the config, load the state dict, eval(), return it.
    ckpt = torch.load(path)
    model = MLP(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

# Step 12 - predict_classes
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict_classes(model, images):
    # TODO: uint8 (n, 28, 28) -> float 0-1 -> standardize -> eval/no_grad -> class names.
    x = torch.tensor(images, dtype=torch.float32) / 255
    x = (x - 0.2860) / 0.3530
    with torch.no_grad():
        model.eval()
        logits = model(x)
        return [CLASS_NAMES[i] for i in logits.argmax(1).tolist()]

