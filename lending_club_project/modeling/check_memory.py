#!/usr/bin/env python3
"""
메모리 사용량 확인 스크립트
"""

import psutil
import torch
import numpy as np

def check_system_resources():
    """시스템 리소스 확인"""
    print("🔍 시스템 리소스 확인")
    print("=" * 40)
    
    # CPU 정보
    print(f"CPU 코어 수: {psutil.cpu_count()}")
    print(f"CPU 사용률: {psutil.cpu_percent()}%")
    
    # 메모리 정보
    memory = psutil.virtual_memory()
    print(f"전체 메모리: {memory.total / (1024**3):.2f} GB")
    print(f"사용 가능한 메모리: {memory.available / (1024**3):.2f} GB")
    print(f"메모리 사용률: {memory.percent}%")
    
    # 디스크 정보
    disk = psutil.disk_usage('/')
    print(f"디스크 사용률: {disk.percent}%")
    
    # PyTorch GPU 정보
    print(f"\n🔍 PyTorch GPU 정보:")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} 메모리: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")

def estimate_tabnet_memory(n_samples, n_features, batch_size=32):
    """TabNet 메모리 사용량 추정"""
    print(f"\n📊 TabNet 메모리 사용량 추정")
    print("=" * 40)
    
    # 기본 메모리 사용량 (대략적 추정)
    base_memory = 0.5  # GB (모델 파라미터)
    data_memory = (n_samples * n_features * 4) / (1024**3)  # float32 기준
    batch_memory = (batch_size * n_features * 4) / (1024**3)
    
    # TabNet 특성별 추가 메모리
    attention_memory = (n_features * 8 * 4) / (1024**3)  # 주의 메커니즘
    gradient_memory = batch_memory * 3  # 그래디언트 저장
    
    total_estimated = base_memory + data_memory + batch_memory + attention_memory + gradient_memory
    
    print(f"데이터 크기: {n_samples} x {n_features}")
    print(f"배치 크기: {batch_size}")
    print(f"추정 메모리 사용량: {total_estimated:.2f} GB")
    
    # 권장사항
    if total_estimated > 8:
        print("⚠️ 메모리 사용량이 높습니다. Cobra 환경 사용을 권장합니다.")
    else:
        print("✅ 로컬 환경에서 실행 가능합니다.")

if __name__ == "__main__":
    check_system_resources()
    
    # TabNet 테스트용 메모리 추정
    estimate_tabnet_memory(n_samples=1000, n_features=20, batch_size=32)
    estimate_tabnet_memory(n_samples=5000, n_features=50, batch_size=64) 