"""Stateless unidirectional GRU and a sample-matched snapshot MLP control."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .mlp import SmallMLP, cpu_runtime, predict_mlp, validation_loss


class SmallGRU(nn.Module):
    def __init__(self, input_features, hidden_size, layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, features):
        # No hidden state is passed or retained: each window starts from zero.
        output, _ = self.gru(features)
        return self.head(self.dropout(output[:, -1, :]))


def make_model(features, config):
    if config.model == "gru":
        return SmallGRU(features, config.gru_hidden_size, config.gru_layers, config.dropout)
    return SmallMLP(features, config.hidden_sizes, config.dropout)


def fit_sequence_model(train, validation, config):
    if not len(train):
        raise ValueError("no eligible TRAIN sequences")
    with cpu_runtime(config.seed):
        model = make_model(train.matrix.shape[1], config)
        train_loader = DataLoader(
            train,
            batch_size=config.train_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            validation, batch_size=config.train_batch_size, shuffle=False, num_workers=0
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        history, best, best_epoch, stale, checkpoint = [], float("inf"), 0, 0, None
        for epoch in range(1, config.epochs + 1):
            model.train()
            total, count = 0.0, 0
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite sequence training loss")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip, error_if_nonfinite=True
                )
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite sequence gradient")
                optimizer.step()
                total += loss.item() * len(y)
                count += len(y)
            observed = validation_loss(model, val_loader)
            if observed is not None and not np.isfinite(observed):
                raise ValueError("nonfinite validation loss")
            history.append(
                {"epoch": epoch, "train_loss": total / count, "validation_loss": observed}
            )
            if observed is None:
                best_epoch = epoch
            elif observed < best - config.min_delta:
                best, best_epoch, stale = observed, epoch, 0
                checkpoint = {
                    name: value.detach().clone() for name, value in model.state_dict().items()
                }
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if checkpoint is not None:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, {
            "status": "fitted",
            "epochs_executed": len(history),
            "selected_epoch": best_epoch,
            "best_validation_loss": best if checkpoint is not None else None,
            "selection": "validation_cross_entropy"
            if len(validation)
            else "fixed_epochs_no_validation",
            "history": history,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "device": "cpu",
            "thread_limit": 1,
            "optimizer": "AdamW",
            "gradient_clip": config.gradient_clip,
            "seed": config.seed,
            "dataloader_seed": config.seed,
            "train_shuffle": True,
            "validation_shuffle": False,
            "dataloader_workers": 0,
            "hidden_state_policy": "zero initialization per independent window",
            "determinism": "seeded deterministic CPU algorithms; "
            "cross-platform/version identity not guaranteed",
        }


def load_sequence_weights(path, features, config):
    with cpu_runtime(config.seed):
        model = make_model(features, config)
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite saved sequence weights")
        model.eval()
        return model


class FrozenSequence:
    def __init__(self, state, preprocessor, estimator):
        self.state, self.preprocessor, self.estimator = state, preprocessor, estimator

    def predict(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        length = self.state["config"]["sequence_length"]
        width = len(self.state["feature_columns"])
        shape = (length, width) if self.state["model"] == "gru" else (width,)
        if matrix.shape[1:] != shape or not np.isfinite(matrix).all():
            raise ValueError("expected a validated transformed sequence/snapshot batch")
        p = predict_mlp(self.estimator, matrix)
        mix = self.state["config"]["probability_mix"]
        p = (1 - mix) * p + mix / 3
        if not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError("invalid sequence probabilities")
        return (p.argmax(axis=1) - 1).astype(np.int8), p
