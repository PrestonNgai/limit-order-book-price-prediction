"""Small seeded CPU MLP with TRAIN minibatches and validation-only checkpoint selection."""

import random
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@contextmanager
def cpu_runtime(seed=None):
    """Restore caller RNG/thread/determinism settings after a bounded CPU operation."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        if threads != 1:
            torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if threads != 1:
            torch.set_num_threads(threads)


class SnapshotDataset(Dataset):
    def __init__(self, matrix, target):
        if matrix.ndim != 2 or len(matrix) != len(target):
            raise ValueError("snapshot/target shapes disagree")
        if not np.isfinite(matrix).all() or not np.isin(target, [-1, 0, 1]).all():
            raise ValueError("invalid MLP inputs")
        self.features = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
        self.target = torch.from_numpy(np.asarray(target, dtype=np.int64) + 1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.features[index], self.target[index]


class SmallMLP(nn.Module):
    def __init__(self, input_features, hidden_sizes, dropout):
        super().__init__()
        layers, previous = [], input_features
        for size in hidden_sizes:
            layers.extend((nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)))
            previous = size
        layers.append(nn.Linear(previous, 3))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)


def validation_loss(model, loader):
    model.eval()
    loss, count = 0.0, 0
    with torch.inference_mode():
        for x, y in loader:
            loss += nn.functional.cross_entropy(model(x), y, reduction="sum").item()
            count += len(y)
    return loss / count if count else None


def fit_mlp(train_x, train_y, validation_x, validation_y, config):
    with cpu_runtime(config.seed):
        model = SmallMLP(train_x.shape[1], config.hidden_sizes, config.dropout)
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(
            SnapshotDataset(train_x, train_y),
            batch_size=config.train_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        validation_loader = DataLoader(
            SnapshotDataset(validation_x, validation_y),
            batch_size=config.train_batch_size,
            shuffle=False,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        history, best, best_epoch, bad_epochs, checkpoint = [], float("inf"), 0, 0, None
        for epoch in range(1, config.epochs + 1):
            model.train()
            training_loss, count = 0.0, 0
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite MLP training loss")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise ValueError("nonfinite MLP gradients")
                optimizer.step()
                training_loss += loss.item() * len(y)
                count += len(y)
            observed = validation_loss(model, validation_loader)
            if observed is not None and not np.isfinite(observed):
                raise ValueError("nonfinite MLP validation loss")
            history.append(
                {"epoch": epoch, "train_loss": training_loss / count, "validation_loss": observed}
            )
            if observed is None:
                best_epoch = epoch
            elif observed < best - config.min_delta:
                best, best_epoch, bad_epochs = observed, epoch, 0
                checkpoint = {
                    name: value.detach().clone() for name, value in model.state_dict().items()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= config.patience:
                    break
        if checkpoint is not None:
            model.load_state_dict(checkpoint)
        model.eval()
        report = {
            "status": "fitted",
            "backend": "torch",
            "device": "cpu",
            "thread_limit": 1,
            "epochs_executed": len(history),
            "selected_epoch": best_epoch,
            "best_validation_loss": best if checkpoint is not None else None,
            "selection": "validation_cross_entropy"
            if len(validation_y)
            else "fixed_epochs_no_validation",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "history": history,
            "optimizer": "AdamW",
            "drop_last": False,
            "dataloader_workers": 0,
            "train_shuffle": True,
            "validation_shuffle": False,
            "seeds": {
                "python": config.seed,
                "numpy": config.seed,
                "torch": config.seed,
                "dataloader": config.seed,
            },
            "determinism": "CPU, one thread, deterministic algorithms; reproducibility across "
            "library versions/platforms is not guaranteed",
        }
        return model, report


def predict_mlp(model, matrix):
    with cpu_runtime(), torch.inference_mode():
        model.eval()
        logits = model(torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32)))
        return torch.softmax(logits.to(torch.float64), dim=1).numpy()


def save_mlp(model, path):
    torch.save(model.state_dict(), path)


def load_mlp(path, input_features, config):
    with cpu_runtime(config.seed):
        model = SmallMLP(input_features, config.hidden_sizes, config.dropout)
        weights = torch.load(path, weights_only=True, map_location="cpu")
        model.load_state_dict(weights, strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite saved MLP weights")
        model.eval()
        return model
