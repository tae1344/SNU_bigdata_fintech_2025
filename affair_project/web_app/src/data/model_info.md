# ONNX 모델 정보 (model.onnx)

## 🎯 **모델 기본 정보**
- **모델 타입**: `RandomForestClassifier`
- **프로듀서**: `skl2onnx` (scikit-learn → ONNX 변환)
- **ONNX 버전**: 1 (opset 1)
- **모델 버전**: 7
- **파일 경로**: `/public/model.onnx`

## 🌳 **RandomForest 상세 사양**
- **트리 개수**: 200개
- **최대 깊이**: 7
- **최소 샘플 리프**: 2
- **최소 샘플 분할**: 5
- **클래스 가중치**: `balanced_subsample`
- **특성 개수**: 12개
- **랜덤 시드**: 42

## 🔍 **입출력 구조**
- **입력**: `float_input` (1×12 차원)
- **출력**: 
  - `label`: 예측 클래스 (0 또는 1)
  - `probabilities`: 클래스별 확률 (1×2 차원)

## 🏆 **모델 성능 등급**
**`modeling_results.json`의 `premium.RandomForest`와 정확히 일치!**

- **F1-Score**: 37.99%
- **ROC-AUC**: 68.45%
- **정확도**: 65.64%
- **정밀도**: 27.99%
- **재현율**: 59.19%

## 🎯 **특성 중요도 순위**
1. **`occupation_husb_grade6`**: 18.41% (남편 직업 등급)
2. **`rating_5`**: 17.74% (결혼 만족도)
3. **`occupation_grade6`**: 17.47% (본인 직업 등급)
4. **`children`**: 10.73% (자녀 수)
5. **`age`**: 7.34% (나이)
6. **`occupation_diff`**: 6.81% (직업 등급 차이)
7. **`yrs_per_age`**: 5.10% (결혼연수/나이 비율)
8. **`gender_male`**: 4.57% (성별)
9. **`rate_x_yrs`**: 3.64% (만족도×결혼연수)
10. **`yearsmarried`**: 3.52% (결혼 연수)
11. **`religiousness_5`**: 2.93% (종교성)
12. **`education`**: 1.73% (교육 수준)

## 📊 **특성 설명**
- **`age`**: 나이 (18-100)
- **`yearsmarried`**: 결혼 연수 (0-80)
- **`children`**: 자녀 수 (0-10)
- **`religiousness_5`**: 종교 활동 빈도 (1-5, 1=무신론적, 5=매우 종교적)
- **`education`**: 교육 수준 (8-20년)
- **`occupation_grade6`**: 본인 직업 등급 (1-6, 1=하위, 6=최상위)
- **`occupation_husb_grade6`**: 남편 직업 등급 (1-6, 1=하위, 6=최상위)
- **`rating_5`**: 결혼 만족도 (1, 3, 5, 1=불만족, 5=매우 만족)
- **`gender_male`**: 성별 (0=여성, 1=남성)
- **`yrs_per_age`**: 결혼연수/나이 비율 (파생 변수)
- **`rate_x_yrs`**: 만족도×결혼연수 (파생 변수)
- **`occupation_diff`**: 직업 등급 차이 (파생 변수)

## 🚀 **사용 방법**
웹 애플리케이션에서 `ONNXModelPredictor` 컴포넌트를 통해 실시간 추론이 가능합니다.

## 📝 **변환 정보**
- **원본 모델**: `src/data/best_model_best_overall_set.pkl`
- **변환 도구**: `skl2onnx`
- **변환 옵션**: `zipmap=False` (표준 텐서 출력)
- **호환성**: `onnxruntime-web` (웹 브라우저)

## 💡 **결론**
이 모델은 최고 품질의 데이터로 훈련된 RandomForest 모델로, 웹 브라우저에서 실시간으로 불륜 위험도를 예측할 수 있습니다.
