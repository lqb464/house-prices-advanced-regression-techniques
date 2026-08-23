"""Khởi tạo model ML truyền thống và huấn luyện GRU."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.rnn import ReturnGRU


def traditional_models(seed: int = 42) -> dict:
    return {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=seed,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def fit_traditional_models(
    frame, feature_columns: list[str], target_column: str, end: int, seed: int = 42
) -> dict:
    """Fit các model ML truyền thống trên phần dữ liệu train đã khóa."""
    estimators = traditional_models(seed)
    for estimator in estimators.values():
        estimator.fit(frame.iloc[:end][feature_columns], frame.iloc[:end][target_column])
    return estimators


def train_gru(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    hidden_size: int = 24,
    epochs: int = 70,
    learning_rate: float = 0.002,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[ReturnGRU, dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = ReturnGRU(x_train.shape[-1], hidden_size)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    loss_function = torch.nn.L1Loss()
    best_state, best_loss, best_epoch, patience = None, float("inf"), 0, 10

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        for batch in order.split(batch_size):
            inputs = torch.from_numpy(x_train[batch])
            labels = torch.from_numpy(y_train[batch])
            optimizer.zero_grad()
            loss = loss_function(model(inputs), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(torch.from_numpy(x_validation)), torch.from_numpy(y_validation)
            ).item()
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_mae": best_loss}


def predict_gru(model: ReturnGRU, sequences: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(sequences)).numpy()
