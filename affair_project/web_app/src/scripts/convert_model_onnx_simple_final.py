import joblib
import numpy as np
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def convert_model_to_onnx_simple_final():
    """간단한 ONNX 모델 변환 (zipmap=False만 사용)"""
    
    print("🔧 간단한 ONNX 모델 변환 시작")
    
    # 모델 파일 경로
    model_path = Path(__file__).parent / "src/data/best_model_best_overall_set.pkl"
    
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return False
    
    try:
        # 모델 로드
        print("📥 모델 로딩 중...")
        model = joblib.load(model_path)
        print(f"✅ 모델 로드 성공: {type(model).__name__}")
        
        # 모델 정보 확인
        print(f"   - 입력 특성 수: {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
            print(f"   - 훈련 시 특성명: {list(model.feature_names_in_)}")
        
        # 특성 순서 정의 (모델 훈련 시와 동일)
        feature_names = [
            'age', 'children', 'religiousness_5', 'education',
            'occupation_grade6', 'occupation_husb_grade6',
            'gender_male', 'gender_female', 'yearsmarried',
            'yrs_per_age', 'rating_5', 'rate_x_yrs'
        ]
        
        print(f"   - 사용할 특성명: {feature_names}")
        print(f"   - 특성 수: {len(feature_names)}")
        
        # 입력 타입 정의
        n_features = len(feature_names)
        initial_type = [('float_input', FloatTensorType([1, n_features]))]
        
        # 변환 옵션 (zipmap=False만 사용)
        options = {
            'zipmap': False,  # 딕셔너리 대신 표준 텐서 출력
        }
        
        print(f"\n🔄 ONNX 변환 중...")
        print(f"   - 옵션: {options}")
        
        # ONNX 모델 변환
        onx = convert_sklearn(
            model, 
            initial_types=initial_type,
            target_opset=12,
            options=options,
            verbose=1
        )
        
        # 모델 저장
        output_path = Path(__file__).parent / "public/model.onnx"
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        print(f"✅ ONNX 모델 저장 완료: {output_path}")
        
        # 변환된 모델 테스트
        print(f"\n🧪 변환된 모델 테스트...")
        
        try:
            import onnxruntime as ort
            
            # ONNX 모델 로드
            session = ort.InferenceSession(str(output_path))
            print(f"   - ONNX 모델 로드 성공")
            print(f"   - 입력 이름: {session.get_inputs()[0].name}")
            print(f"   - 출력 이름: {[output.name for output in session.get_outputs()]}")
            
            # 테스트 데이터 생성
            test_input = np.array([
                30, 2, 3, 12, 3, 3, 0, 1, 5, 5/30, 5, 5*5
            ], dtype=np.float32).reshape(1, -1)
            
            print(f"   - 테스트 입력 형태: {test_input.shape}")
            
            # 모델 실행
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: test_input})
            
            print(f"   - 출력 개수: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"   - 출력 {i}: 형태={output.shape}, 타입={output.dtype}")
                if output.size <= 10:  # 작은 출력만 내용 표시
                    print(f"     내용: {output}")
            
            print(f"✅ ONNX 모델 테스트 성공!")
            return True
            
        except Exception as e:
            print(f"❌ ONNX 모델 테스트 실패: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 모델 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_model_to_onnx_simple_final()
    if success:
        print("\n🎉 ONNX 모델 변환 완료!")
    else:
        print("\n💥 ONNX 모델 변환 실패!")
