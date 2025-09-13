import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys
import joblib

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 데이터 로드
data = pd.read_parquet(project_root / "data/processed/dataset.parquet")

# 타깃: adr, 이상치 제거(음수/0 체류)
data = data[(data["stay_nights"] > 0) & (data["adr"] > 0)].copy()

# 타깃 분리
y = data["adr"].values
X = data.drop(columns=["is_canceled", "arrival_date"])

# 특성 분리
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

pre = ColumnTransformer(
    [
        ("cats", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("nums", "passthrough", num_cols),
    ]
)

# 모델 정의
reg = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    boosting_type="gbdt",
    max_depth=-1, # 트리 깊이
    num_leaves=63, # 리프 노드 수
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# 파이프라인 정의
pipe = Pipeline([("pre", pre), ("reg", reg)])
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_tr, y_tr)

# 예측
pred = pipe.predict(X_va)
rmse = mean_squared_error(y_va, pred)   
print("RMSE:", rmse)

# RMSLE (스케일 민감도↓, 비율 오차 성격) — 음수 방지 위해 예측값을 0으로 클리핑
pred_clip = np.clip(pred, 0, None)
rmsle = np.sqrt(mean_squared_log_error(y_va, pred_clip))
print("RMSLE:", rmsle)

# 아티팩트 저장
ART = Path(os.path.join(project_root, "artifacts"))
ART.mkdir(parents=True, exist_ok=True)

np.save(ART / "reg_valid_pred.npy", pred.astype(np.float32))
np.save(ART / "reg_valid_true.npy", y_va.astype(np.float32))

# 파이프라인 저장
joblib.dump(pipe, ART / "lgbm_adr_pipeline.joblib")

print(f"[Saved] {ART}/reg_valid_pred.npy, reg_valid_true.npy")
