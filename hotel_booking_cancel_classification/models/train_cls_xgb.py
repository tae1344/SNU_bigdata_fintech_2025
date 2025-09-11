import pandas as pd, numpy as np, xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    classification_report,
)
from pathlib import Path
import os
import sys
import joblib

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 데이터 로드
data = pd.read_parquet(project_root / "data/processed/dataset.parquet")

# 시간 기반 스플릿
train = data[data["arrival_date"] < "2017-01-01"].copy()
valid = data[data["arrival_date"] >= "2017-01-01"].copy()

# 타깃 분리
y_tr, y_va = train.pop("is_canceled").values, valid.pop("is_canceled").values

# 날짜 컬럼 제거
train.drop(columns=["arrival_date"], inplace=True)
valid.drop(columns=["arrival_date"], inplace=True)

# 특성 분리
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
num_cols = train.select_dtypes(exclude=["object"]).columns.tolist()

pre = ColumnTransformer(
    [
        ("cats", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("nums", "passthrough", num_cols),
    ]
)

# 모델 정의
clf = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
)

# 파이프라인 정의
pipe = Pipeline([("pre", pre), ("clf", clf)])
pipe.fit(train, y_tr)

proba = pipe.predict_proba(valid)[:, 1]
pred = (proba >= 0.5).astype(int)

# 성능 평가
print("AUC:", roc_auc_score(y_va, proba))
print("PR-AUC:", average_precision_score(y_va, proba))
print("F1(macro):", f1_score(y_va, pred, average="macro"))
print(classification_report(y_va, pred, digits=3))

# 아티팩트 저장
ART = Path(os.path.join(project_root, "artifacts"))
ART.mkdir(parents=True, exist_ok=True)

np.save(ART / "valid_proba.npy", proba.astype(np.float32))
np.save(ART / "valid_pred.npy", pred.astype(np.int8))
np.save(ART / "valid_true.npy", y_va.astype(np.int8))

# 파이프라인 저장
joblib.dump(pipe, ART / "xgb_cancel_pipeline.joblib")

# Booster (원 모델) 저장
pipe.named_steps["clf"].get_booster().save_model(str(ART / "xgb_cancel_model.json"))

print(f"[Saved] {ART}/valid_proba.npy, valid_pred.npy, valid_true.npy")
