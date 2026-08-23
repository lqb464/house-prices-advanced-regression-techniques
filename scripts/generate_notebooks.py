"""Tạo năm notebook DS/MLE chuẩn cho StocKast."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.write(notebook, NOTEBOOKS / name)


COMMON = r"""
import json
import os
import warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
MPL_CONFIG = ROOT / ".matplotlib"
MPL_CONFIG.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# Run minh hoạ mặc định dùng VIC; pipeline nhận ticker khác qua config/CLI.
DATA_PATH = ROOT / "data" / "raw" / "VIC_VN.csv"
REPORT_DIR = ROOT / "evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)
"""


FEATURE_FUNCTION = r"""
FEATURES = [
    "return_1d", "return_5d", "volatility_10", "volatility_20",
    "volume_z_20", "rsi_14", "macd_scaled", "price_ma20", "price_ma50",
]
TARGET = "target_return_5d"

def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    returns = np.log(close / close.shift(1))
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["return_1d"] = returns
    df["return_5d"] = np.log(close / close.shift(5))
    df["volatility_10"] = returns.rolling(10).std()
    df["volatility_20"] = returns.rolling(20).std()
    df["volume_z_20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    df["rsi_14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) / 100
    df["macd_scaled"] = macd / close
    df["price_ma20"] = close / close.rolling(20).mean() - 1
    df["price_ma50"] = close / close.rolling(50).mean() - 1
    df[TARGET] = np.log(close.shift(-5) / close)
    return df.replace([np.inf, -np.inf], np.nan)
"""


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    write(
        "01_EDA.ipynb",
        [
            markdown("""
# 01 — Phân tích dữ liệu khám phá

## Định nghĩa bài toán nghiệp vụ

- **Quyết định:** hỗ trợ chuyên viên phân tích cổ phiếu rà soát kịch bản return 5 phiên sau khi thị trường đóng cửa; output không phải giao dịch tự động hay price target.
- **Đơn vị dự báo:** một cặp ticker-date. Model dự báo cumulative log-return từ giá đóng cửa ngày `t` đến giá đóng cửa ngày `t+5`.
- **Population:** dữ liệu OHLCV ngày của cổ phiếu niêm yết thanh khoản, có ít nhất 500 phiên đúng thứ tự. Run minh hoạ kiểm chứng workflow trên ticker cấu hình (hiện là VIC, mã provider `VIC.VN`) từ năm 2018.
- **Chi phí sai lệch:** overprediction có thể tạo long exposure không cần thiết, vì vậy directional false positive được theo dõi cùng MAE đối xứng. Underprediction chủ yếu thể hiện phần tăng giá bị bỏ lỡ.
- **Baseline:** zero-return và return 5 phiên gần nhất.
- **Input bị loại:** định danh date/ticker và mọi giá trị quan sát sau khi đóng cửa ngày `t`. OHLCV là dữ liệu thị trường, không phải thuộc tính cá nhân nhạy cảm.

Dataset chỉ hỗ trợ dự báo return lịch sử và backtesting. Dataset không chứa ý định giao dịch, ràng buộc danh mục, fundamentals hay label chứng minh mức độ phù hợp đầu tư.
"""),
            code(COMMON),
            code(r"""
raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
required = ["date", "open", "high", "low", "close", "volume"]
audit = {
    "rows": len(raw),
    "start": raw["date"].min().date().isoformat(),
    "end": raw["date"].max().date().isoformat(),
    "duplicate_dates": int(raw["date"].duplicated().sum()),
    "missing_cells": int(raw[required].isna().sum().sum()),
    "non_positive_prices": int((raw[["open", "high", "low", "close"]] <= 0).sum().sum()),
    "negative_volume": int((raw["volume"] < 0).sum()),
}
audit
"""),
            code(r"""
returns = np.log(raw["close"] / raw["close"].shift(1))
target_5d = np.log(raw["close"].shift(-5) / raw["close"])
summary = pd.DataFrame({"close": raw["close"], "return_1d": returns, "target_return_5d": target_5d}).describe().T
summary
"""),
            code(r"""
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(raw["date"], raw["close"], color="#2563eb")
axes[0].set(title="Giá đóng cửa đã điều chỉnh của ticker ví dụ (VIC)", ylabel="VND")
axes[1].hist(returns.dropna() * 100, bins=60, color="#0f766e", alpha=.85)
axes[1].set(title="Phân phối log-return ngày", xlabel="Log-return (%)", ylabel="Số phiên")
plt.tight_layout()
plt.show()
"""),
            code(r"""
q1, q3 = returns.quantile([.25, .75])
iqr = q3 - q1
outliers = int(((returns < q1 - 3 * iqr) | (returns > q3 + 3 * iqr)).sum())
conclusion = {
    "target_choice": "cumulative log-return 5 phiên; giá đóng cửa thô không dừng và không phù hợp làm target trực tiếp",
    "outlier_sessions_3iqr": outliers,
    "leakage_action": "tạo target từ giá đóng cửa tương lai, sau đó loại target/date khỏi production feature",
    "population_action": "từ chối chuỗi sai thứ tự, trùng lặp, giá không dương hoặc ngắn hơn 500 phiên",
}
print(json.dumps(conclusion, indent=2))
"""),
        ],
    )

    write(
        "02_Feature_Engineering.ipynb",
        [
            markdown("""
# 02 — Feature Engineering

Feature chỉ được tính từ thông tin có sẵn tại hoặc trước khi đóng cửa ngày `t`. Ranh giới train/validation/holdout theo thời gian được tạo trước khi fit bất kỳ scaler nào. `date` được giữ để audit và không bao giờ đi vào production matrix.
"""),
            code(COMMON + "\n" + FEATURE_FUNCTION),
            code(r"""
raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
frame = build_frame(raw).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
n = len(frame)
train_end, validation_end = int(n * .60), int(n * .80)
train = frame.iloc[:train_end].copy()
validation = frame.iloc[train_end:validation_end].copy()
holdout = frame.iloc[validation_end:].copy()
split_summary = pd.DataFrame({
    "rows": [len(train), len(validation), len(holdout)],
    "start": [x.date.min().date() for x in (train, validation, holdout)],
    "end": [x.date.max().date() for x in (train, validation, holdout)],
}, index=["train", "validation", "holdout"])
split_summary
"""),
            code(r"""
corr = train[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
matrix_corr = train[FEATURES].corr().abs()
high_pairs = [
    (a, b, round(float(matrix_corr.loc[a, b]), 3))
    for i, a in enumerate(FEATURES)
    for b in FEATURES[i + 1:]
    if matrix_corr.loc[a, b] > .90
]
print("Tương quan feature-target (chỉ trên train):")
display(corr.to_frame("correlation"))
print("Các cặp vượt |0,90|:", high_pairs)
"""),
            code(r"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train[FEATURES])
scaled_train = scaler.transform(train[FEATURES])
assert np.isfinite(scaled_train).all()
feature_contract = {
    "production_features": FEATURES,
    "audit_only": ["date", "open", "high", "low", "close", "volume"],
    "excluded": [TARGET, "ticker/id", "giá đóng cửa tương lai"],
    "sequence_length": 20,
    "train_only_preprocessing": True,
    "split_rows": {"train": len(train), "validation": len(validation), "holdout": len(holdout)},
}
print(json.dumps(feature_contract, indent=2))
"""),
            code(r"""
fig, ax = plt.subplots(figsize=(9, 4))
corr.sort_values().plot.barh(ax=ax, color="#7c3aed")
ax.set(title="Tương quan feature với target 5 phiên, chỉ trên train", xlabel="Tương quan Pearson")
plt.tight_layout()
plt.show()
"""),
        ],
    )

    write(
        "03_Model_Experiments.ipynb",
        [
            markdown("""
# 03 — Thí nghiệm model và hybrid ensemble

Notebook so sánh baseline nghiệp vụ/thống kê, linear regression có regularization, hai tree ensemble và RNN thực sự với chuỗi 20 phiên. Model truyền thống dùng dòng feature tại `t`; RNN nhận một chuỗi theo thời gian. Hybrid ensemble kết hợp các component đã fit để giảm phụ thuộc vào một họ model. Việc chọn model dựa trên validation MAE và expanding-window CV. Holdout chỉ được đánh giá một lần sau khi khóa model chiến thắng. VIC chỉ là ticker ví dụ được cấu hình cho lần chạy này.
"""),
            code(COMMON + "\n" + FEATURE_FUNCTION),
            code(r"""
from copy import deepcopy
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
from torch import nn

torch.manual_seed(SEED)
torch.set_num_threads(1)

raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
frame = build_frame(raw).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
n = len(frame)
train_end, validation_end = int(n * .60), int(n * .80)
train, validation, holdout = frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]

models = {
    "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
    "extra_trees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=8, max_features=.8, random_state=SEED, n_jobs=1),
    "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, learning_rate=.04, l2_regularization=1.0, random_state=SEED),
}

def metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** .5),
        "directional_accuracy": float(np.mean(np.sign(y) == np.sign(pred))),
    }
"""),
            code(r"""
cv = TimeSeriesSplit(n_splits=4, gap=5)
cv_rows = []
combined = pd.concat([train, validation]).reset_index(drop=True)
for name, estimator in models.items():
    fold_mae = []
    for fold, (fit_idx, score_idx) in enumerate(cv.split(combined), 1):
        fitted = clone(estimator).fit(combined.loc[fit_idx, FEATURES], combined.loc[fit_idx, TARGET])
        pred = fitted.predict(combined.loc[score_idx, FEATURES])
        fold_mae.append(mean_absolute_error(combined.loc[score_idx, TARGET], pred))
    cv_rows.append({"model": name, "cv_mae_mean": np.mean(fold_mae), "cv_mae_std": np.std(fold_mae)})
cv_results = pd.DataFrame(cv_rows).sort_values("cv_mae_mean")
cv_results
"""),
            code(r"""
validation_predictions = {
    "zero_return": np.zeros(len(validation)),
    "recent_5d_return": validation["return_5d"].to_numpy(),
}
fitted_models = {}
for name, estimator in models.items():
    fitted_models[name] = clone(estimator).fit(train[FEATURES], train[TARGET])
    validation_predictions[name] = fitted_models[name].predict(validation[FEATURES])

validation_rows = [
    {"model": name, **metrics(validation[TARGET], pred)}
    for name, pred in validation_predictions.items()
]
pd.DataFrame(validation_rows).sort_values("mae")
"""),
            code(r"""
SEQ_LEN = 20

def sequence_split(frame, start, end, scaler):
    values = scaler.transform(frame[FEATURES]).astype("float32")
    targets = frame[TARGET].to_numpy(dtype="float32")
    xs, ys, positions = [], [], []
    for pos in range(max(SEQ_LEN - 1, start), end):
        xs.append(values[pos - SEQ_LEN + 1: pos + 1])
        ys.append(targets[pos])
        positions.append(pos)
    return np.stack(xs), np.asarray(ys), np.asarray(positions)

class ReturnRNN(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.rnn = nn.GRU(inputs, hidden_size=24, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(24, 1))
    def forward(self, x):
        output, _ = self.rnn(x)
        return self.head(output[:, -1]).squeeze(1)

def train_rnn(frame, train_end, validation_end, epochs=70):
    scaler = StandardScaler().fit(frame.iloc[:train_end][FEATURES])
    x_train, y_train, _ = sequence_split(frame, 0, train_end, scaler)
    x_val, y_val, val_pos = sequence_split(frame, train_end, validation_end, scaler)
    net = ReturnRNN(len(FEATURES))
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    best, patience = None, 10
    for epoch in range(epochs):
        net.train(); order = torch.randperm(len(x_train))
        for batch in order.split(64):
            xb, yb = torch.from_numpy(x_train[batch]), torch.from_numpy(y_train[batch])
            optimizer.zero_grad(); loss = loss_fn(net(xb), yb); loss.backward(); optimizer.step()
        net.eval()
        with torch.no_grad(): val_loss = loss_fn(net(torch.from_numpy(x_val)), torch.from_numpy(y_val)).item()
        if best is None or val_loss < best[0] - 1e-6:
            best = (val_loss, deepcopy(net.state_dict()), epoch + 1); patience = 10
        else:
            patience -= 1
            if patience == 0: break
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): val_pred = net(torch.from_numpy(x_val)).numpy()
    return net, scaler, val_pred, val_pos, best[2]

rnn, rnn_scaler, rnn_val_pred, rnn_val_pos, best_epoch = train_rnn(frame, train_end, validation_end)
validation_predictions["gru_rnn"] = rnn_val_pred
rnn_validation_metrics = metrics(frame.loc[rnn_val_pos, TARGET], rnn_val_pred)
ensemble_components = ["ridge", "extra_trees", "hist_gradient_boosting", "gru_rnn"]
validation_predictions["hybrid_ensemble"] = np.mean(
    [validation_predictions[name] for name in ensemble_components], axis=0
)
ensemble_validation_metrics = metrics(
    frame.loc[rnn_val_pos, TARGET], validation_predictions["hybrid_ensemble"]
)
rnn_validation_metrics, ensemble_validation_metrics, best_epoch
"""),
            code(r"""
selection = pd.DataFrame(
    validation_rows
    + [
        {"model": "gru_rnn", **rnn_validation_metrics},
        {"model": "hybrid_ensemble", **ensemble_validation_metrics},
    ]
)
selection = selection.merge(cv_results, on="model", how="left")
selection["eligible"] = selection["model"].isin(
    ["ridge", "extra_trees", "hist_gradient_boosting", "gru_rnn", "hybrid_ensemble"]
)
winner = selection[selection.eligible].sort_values(["mae", "cv_mae_std"], na_position="last").iloc[0]["model"]
print(selection.sort_values("mae").to_string(index=False))
print("MODEL ĐÃ KHÓA TRƯỚC KHI MỞ HOLDOUT:", winner)
"""),
            code(r"""
# Holdout được dùng lần đầu và duy nhất sau khi `winner` đã được khóa ở trên.
if winner in {"gru_rnn", "hybrid_ensemble"}:
    final_rnn, final_scaler, _, _, _ = train_rnn(frame, validation_end, validation_end + 1, epochs=max(25, best_epoch + 5))
    x_hold, y_hold, hold_pos = sequence_split(frame, validation_end, len(frame), final_scaler)
    with torch.no_grad(): hold_pred = final_rnn(torch.from_numpy(x_hold)).numpy()
    hold_dates = frame.loc[hold_pos, "date"].to_numpy()
    if winner == "hybrid_ensemble":
        final_traditional = {}
        for name, estimator in models.items():
            final_traditional[name] = clone(estimator).fit(
                frame.iloc[:validation_end][FEATURES], frame.iloc[:validation_end][TARGET]
            )
        traditional_hold = np.column_stack(
            [final_traditional[name].predict(frame.loc[hold_pos, FEATURES]) for name in models]
        )
        hold_pred = np.mean(np.column_stack([traditional_hold, hold_pred]), axis=1)
else:
    final_model = clone(models[winner]).fit(frame.iloc[:validation_end][FEATURES], frame.iloc[:validation_end][TARGET])
    hold_pred = final_model.predict(holdout[FEATURES])
    y_hold = holdout[TARGET].to_numpy()
    hold_dates = holdout["date"].to_numpy()

holdout_metrics = metrics(y_hold, hold_pred)
experiment = {
    "target": TARGET,
    "horizon_sessions": 5,
    "selected_model": winner,
    "selection_rule": "validation MAE thấp nhất trong các model có thể train; độ ổn định CV dùng để phân xử khi bằng nhau",
    "validation": selection.fillna(value={"cv_mae_mean": -1, "cv_mae_std": -1}).to_dict("records"),
    "holdout": holdout_metrics,
    "holdout_rows": int(len(y_hold)),
    "holdout_start": str(pd.Timestamp(hold_dates[0]).date()),
    "holdout_end": str(pd.Timestamp(hold_dates[-1]).date()),
    "rnn_sequence_length": SEQ_LEN,
    "rnn_best_epoch": int(best_epoch),
}
(REPORT_DIR / "model_metrics.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
pd.DataFrame({"date": hold_dates, "actual": y_hold, "prediction": hold_pred}).to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)
print(json.dumps(experiment, indent=2))
"""),
        ],
    )

    write(
        "04_Model_Interpretation.ipynb",
        [
            markdown("""
# 04 — Diễn giải model và phân tích lỗi

Việc diễn giải không làm thay đổi model đã chọn. Permutation importance được tính trên validation cho candidate ML truyền thống tốt nhất, còn dự báo trên holdout chỉ dùng để phân tích lỗi sau lựa chọn.
"""),
            code(COMMON + "\n" + FEATURE_FUNCTION),
            code(r"""
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
frame = build_frame(raw).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
n = len(frame); train_end, validation_end = int(n * .60), int(n * .80)
train, validation = frame.iloc[:train_end], frame.iloc[train_end:validation_end]
metrics_report = json.loads((REPORT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
traditional_rows = [x for x in metrics_report["validation"] if x["model"] in {"ridge", "extra_trees", "hist_gradient_boosting"}]
interpreted_name = min(traditional_rows, key=lambda x: x["mae"])["model"]
estimators = {
    "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
    "extra_trees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=8, max_features=.8, random_state=SEED, n_jobs=1),
    "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, learning_rate=.04, l2_regularization=1.0, random_state=SEED),
}
interpreted = estimators[interpreted_name].fit(train[FEATURES], train[TARGET])
perm = permutation_importance(interpreted, validation[FEATURES], validation[TARGET], scoring="neg_mean_absolute_error", n_repeats=15, random_state=SEED)
importance = pd.DataFrame({"feature": FEATURES, "mae_increase": perm.importances_mean, "std": perm.importances_std}).sort_values("mae_increase", ascending=False)
print("Model truyền thống được diễn giải:", interpreted_name)
importance
"""),
            code(r"""
preds = pd.read_csv(REPORT_DIR / "holdout_predictions.csv", parse_dates=["date"])
preds["error"] = preds["prediction"] - preds["actual"]
preds["absolute_error"] = preds["error"].abs()
preds["actual_band"] = pd.qcut(preds["actual"], 3, labels=["down", "flat", "up"])
preds["volatility_regime"] = pd.qcut(preds["actual"].abs(), 3, labels=["low", "medium", "high"])
error_by_band = preds.groupby("actual_band", observed=True).agg(rows=("actual", "size"), mae=("absolute_error", "mean"), bias=("error", "mean"))
error_by_volatility = preds.groupby("volatility_regime", observed=True).agg(rows=("actual", "size"), mae=("absolute_error", "mean"))
display(error_by_band)
display(error_by_volatility)
"""),
            code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
importance.sort_values("mae_increase").plot.barh(x="feature", y="mae_increase", ax=axes[0], legend=False, color="#2563eb")
axes[0].set(title=f"Permutation importance — {interpreted_name}", xlabel="Mức tăng validation MAE")
axes[1].scatter(preds["actual"] * 100, preds["prediction"] * 100, alpha=.55, color="#0f766e")
limit = float(max(preds.actual.abs().max(), preds.prediction.abs().max()) * 100)
axes[1].plot([-limit, limit], [-limit, limit], "--", color="gray")
axes[1].set(title="Model đã khóa trên holdout", xlabel="Return 5 ngày thực tế (%)", ylabel="Dự báo (%)")
plt.tight_layout(); plt.show()
"""),
            code(r"""
worst = preds.nlargest(5, "absolute_error")[["date", "actual", "prediction", "absolute_error"]]
interpretation = {
    "selected_model": metrics_report["selected_model"],
    "interpreted_proxy": interpreted_name,
    "top_features": importance.head(3)["feature"].tolist(),
    "highest_error_regime": error_by_volatility["mae"].idxmax(),
    "limitation": "chỉ technical feature không giải thích được biến động mạnh do sự kiện; không khẳng định quan hệ nhân quả hay mức độ phù hợp đầu tư",
}
display(worst)
print(json.dumps(interpretation, indent=2))
"""),
        ],
    )

    write(
        "05_Business_Report.ipynb",
        [
            markdown("""
# 05 — Báo cáo nghiệp vụ và quality gate

Báo cáo chuyển thí nghiệm đã khóa thành các quy tắc hỗ trợ quyết định. Prediction interval và ngưỡng rà soát được calibration từ residual trên validation, không bao giờ từ residual trên holdout.
"""),
            code(COMMON + "\n" + FEATURE_FUNCTION),
            code(r"""
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
frame = build_frame(raw).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
n = len(frame); train_end, validation_end = int(n * .60), int(n * .80)
train, validation = frame.iloc[:train_end], frame.iloc[train_end:validation_end]
report = json.loads((REPORT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
selected = report["selected_model"]
estimators = {
    "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
    "extra_trees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=8, max_features=.8, random_state=SEED, n_jobs=-1),
    "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, learning_rate=.04, l2_regularization=1.0, random_state=SEED),
}
calibration_model_name = selected if selected in estimators else min(
    (x for x in report["validation"] if x["model"] in estimators), key=lambda x: x["mae"]
)["model"]
calibration_model = clone(estimators[calibration_model_name]).fit(train[FEATURES], train[TARGET])
validation_pred = calibration_model.predict(validation[FEATURES])
absolute_residual = np.abs(validation[TARGET].to_numpy() - validation_pred)
half_width = float(np.quantile(absolute_residual, .80))
half_width
"""),
            code(r"""
holdout = pd.read_csv(REPORT_DIR / "holdout_predictions.csv", parse_dates=["date"])
holdout["low"] = holdout["prediction"] - half_width
holdout["high"] = holdout["prediction"] + half_width
holdout["covered"] = holdout["actual"].between(holdout["low"], holdout["high"])
holdout["manual_review"] = holdout["low"].le(0) & holdout["high"].ge(0)
coverage = float(holdout["covered"].mean())
false_positive_rate = float(((holdout["prediction"] > 0) & (holdout["actual"] <= 0)).mean())
zero_return_baseline_mae = float(holdout["actual"].abs().mean())
deployment_reasons = []
if report["holdout"]["mae"] > zero_return_baseline_mae:
    deployment_reasons.append("MAE trên holdout không tốt hơn zero-return baseline")
if coverage < .70:
    deployment_reasons.append("interval coverage trên holdout thấp hơn ngưỡng triển khai 70%")
business_metrics = {
    "technical": report["holdout"],
    "zero_return_baseline_mae": zero_return_baseline_mae,
    "interval_coverage_80": coverage,
    "interval_half_width_log_return": half_width,
    "directional_false_positive_rate": false_positive_rate,
    "manual_review_rate": float(holdout["manual_review"].mean()),
}
business_metrics
"""),
            code(r"""
production_report = {
    "project": "StocKast",
    "ticker": "VIC.VN",
    "scope": "framework dự báo stock time series; report này là run minh hoạ trên VIC",
    "role": "framework Data Scientist và ML Engineer",
    "target": "cumulative log-return 5 phiên sau khi thị trường đóng cửa",
    "selected_model": selected,
    "calibration_model": calibration_model_name,
    "business_baseline": "zero-return",
    "deployment_status": "deployment_blocked" if deployment_reasons else "deployment_ready",
    "deployment_reasons": deployment_reasons,
    "holdout_period": [report["holdout_start"], report["holdout_end"]],
    "metrics": business_metrics,
    "decision_policy": {
        "usage": "chỉ dùng cho kịch bản nghiên cứu và model monitoring",
        "manual_review": "gắn cờ interval cắt qua 0, input ngoài phân phối hoặc cảnh báo schema",
        "business_cap": "clip log-return của model trong +/-0,20; giữ dự báo thô để audit",
        "automation": "không tự động đặt lệnh, phân bổ vốn hay phê duyệt đầu tư",
    },
    "monitoring": ["tính hợp lệ của schema", "feature PSI", "prediction drift", "realized MAE", "interval coverage"],
    "limitations": [
        "run minh hoạ trên một ticker không chứng minh khả năng tổng quát; cần tái lập theo ticker hoặc universe mục tiêu",
        "technical feature từ OHLCV không chứa fundamentals và bối cảnh sự kiện",
        "backtest không chứng minh lợi nhuận tương lai",
    ],
}
(REPORT_DIR / "production_report.json").write_text(json.dumps(production_report, indent=2), encoding="utf-8")
print(json.dumps(production_report, indent=2))
"""),
            code(r"""
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(holdout["date"], holdout["actual"] * 100, label="Thực tế", color="#0f172a", alpha=.7)
ax.plot(holdout["date"], holdout["prediction"] * 100, label="Dự báo", color="#2563eb")
ax.fill_between(holdout["date"], holdout["low"] * 100, holdout["high"] * 100, alpha=.18, color="#2563eb", label="Interval 80% calibration trên validation")
ax.set(title="Kịch bản dự báo trên holdout", ylabel="Log-return 5 phiên (%)", xlabel="Ngày dự báo")
ax.legend(ncol=3); plt.tight_layout(); plt.show()
"""),
        ],
    )


if __name__ == "__main__":
    main()
