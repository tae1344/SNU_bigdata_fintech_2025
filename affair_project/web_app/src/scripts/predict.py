# src/data/predict.py
import sys
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def load_model():
    """훈련된 모델 로드"""
    model_path = Path(__file__).parent / "best_model_best_overall_set.pkl"
    return joblib.load(model_path)

def preprocess_input(user_input):
    """사용자 입력을 모델 입력 형식으로 변환"""
    # 기본 특성
    features = {
        'age': user_input['age'],
        'yearsmarried': user_input['yearsmarried'],
        'children': user_input['children'],
        'religiousness_5': user_input['religiousness_5'],
        'education': user_input['education'],
        'occupation_grade6': user_input['occupation_grade6'],
        'occupation_husb_grade6': user_input['occupation_husb_grade6'],
        'rating_5': user_input['rating_5']
    }
    
    # 성별 원-핫 인코딩
    features['gender_male'] = 1 if user_input['gender'] == 'male' else 0
    features['gender_female'] = 1 if user_input['gender'] == 'female' else 0
    
    # 파생 특성
    features['yrs_per_age'] = user_input['yearsmarried'] / user_input['age']
    features['rate_x_yrs'] = user_input['rating_5'] * user_input['yearsmarried']
    
    # 가중치 (기본값 1.0)
    features['weight'] = 1.0
    
    # 특성 순서 맞추기 (모델 훈련 시 사용된 순서와 동일해야 함)
    feature_order = [
        'age', 'yearsmarried', 'children', 'religiousness_5', 'education',
        'occupation_grade6', 'occupation_husb_grade6', 'rating_5',
        'gender_male', 'gender_female', 'yrs_per_age', 'rate_x_yrs', 'weight'
    ]
    
    return pd.DataFrame([features])[feature_order]

def predict(user_input):
    """모델을 사용하여 예측 수행"""
    try:
        # 모델 로드
        model = load_model()
        
        # 입력 전처리
        X = preprocess_input(user_input)
        
        # 예측 수행
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            probability = proba[1] * 100  # 불륜 확률 (백분율)
        else:
            prediction = model.predict(X)[0]
            probability = prediction * 100
        
        # 위험도 판정
        if probability < 20:
            risk_level = 'low'
        elif probability < 35:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        # 주요 요인 분석
        factors = []
        if user_input['rating_5'] == 1:
            factors.append('낮은 결혼 만족도')
        if user_input['yearsmarried'] >= 20:
            factors.append('긴 결혼 기간')
        if user_input['religiousness_5'] <= 2:
            factors.append('낮은 종교성')
        if user_input['occupation_grade6'] <= 2:
            factors.append('낮은 직업 등급')
        if user_input['children'] == 0:
            factors.append('자녀 없음')
        
        yrs_per_age = user_input['yearsmarried'] / user_input['age']
        if yrs_per_age > 0.7:
            factors.append('일찍 결혼')
        
        # 권장사항
        recommendations = []
        if user_input['rating_5'] == 1:
            recommendations.append('결혼 상담 프로그램 참여')
        if user_input['religiousness_5'] <= 2:
            recommendations.append('종교 활동 참여 고려')
        if user_input['children'] == 0:
            recommendations.append('가족 관계 강화')
        if user_input['occupation_grade6'] <= 2:
            recommendations.append('직업 개발 프로그램')
        
        return {
            'probability': round(probability, 1),
            'riskLevel': risk_level,
            'factors': factors,
            'recommendations': recommendations,
            'model_confidence': 'high'  # 실제 모델 사용
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'probability': 0,
            'riskLevel': 'unknown',
            'factors': [],
            'recommendations': []
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Invalid arguments'}))
        sys.exit(1)
    
    try:
        user_input = json.loads(sys.argv[1])
        result = predict(user_input)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)