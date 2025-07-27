"""
데이터 샘플 생성 스크립트
전체 데이터셋에서 일부 샘플을 추출하여 GitHub에 업로드할 수 있도록 함
"""

import pandas as pd
import numpy as np
import os

def create_data_sample(input_file='lending_club_2020_train.csv', 
                      output_file='lending_club_sample.csv',
                      sample_size=1000,
                      random_state=42):
    """
    전체 데이터셋에서 샘플을 추출하여 저장
    
    Parameters:
    -----------
    input_file : str
        입력 CSV 파일 경로
    output_file : str
        출력 CSV 파일 경로
    sample_size : int
        샘플 크기
    random_state : int
        랜덤 시드
    """
    
    print(f"데이터 샘플 생성 중... (크기: {sample_size})")
    
    # 데이터 로드 (청크 단위로 처리)
    chunk_size = 10000
    sample_chunks = []
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # 각 청크에서 일부 샘플 추출
        chunk_sample = chunk.sample(n=min(sample_size//10, len(chunk)), 
                                  random_state=random_state)
        sample_chunks.append(chunk_sample)
        
        if len(sample_chunks) * (sample_size//10) >= sample_size:
            break
    
    # 모든 샘플을 결합
    sample_df = pd.concat(sample_chunks, ignore_index=True)
    
    # 최종 샘플 크기 조정
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(n=sample_size, random_state=random_state)
    
    # 샘플 저장
    sample_df.to_csv(output_file, index=False)
    
    print(f"샘플 데이터 저장 완료: {output_file}")
    print(f"샘플 크기: {len(sample_df)}행 × {len(sample_df.columns)}열")
    print(f"파일 크기: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return sample_df

def create_data_info_file(sample_df, output_file='data_info.txt'):
    """
    데이터 정보를 텍스트 파일로 저장
    
    Parameters:
    -----------
    sample_df : pandas.DataFrame
        샘플 데이터프레임
    output_file : str
        출력 파일 경로
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Lending Club 데이터셋 정보\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"전체 데이터셋 크기: 1,755,295행 × 141열\n")
        f.write(f"샘플 데이터 크기: {len(sample_df)}행 × {len(sample_df.columns)}열\n")
        f.write(f"메모리 사용량: 약 4.4GB (전체 데이터셋)\n\n")
        
        f.write("주요 변수 카테고리:\n")
        f.write("- 대출자 정보 (11개 변수)\n")
        f.write("- 대출 정보 (15개 변수)\n")
        f.write("- 신용 정보 (8개 변수)\n")
        f.write("- 지급 정보 (11개 변수)\n")
        f.write("- 연체 정보 (12개 변수)\n")
        f.write("- 계좌 정보 (20개 변수)\n")
        f.write("- 잔액 정보 (15개 변수)\n")
        f.write("- 조회 정보 (4개 변수)\n")
        f.write("- 기타 정보 (45개 변수)\n\n")
        
        f.write("종속변수 (loan_status) 분포:\n")
        status_counts = sample_df['loan_status'].value_counts()
        for status, count in status_counts.items():
            percentage = (count / len(sample_df)) * 100
            f.write(f"- {status}: {count}개 ({percentage:.2f}%)\n")
        
        f.write("\n데이터 다운로드 방법:\n")
        f.write("1. Kaggle에서 'Lending Club Dataset' 검색\n")
        f.write("2. 'Lending Club 2020' 데이터셋 다운로드\n")
        f.write("3. lending_club_2020_train.csv 파일을 프로젝트 루트에 저장\n")
    
    print(f"데이터 정보 파일 저장 완료: {output_file}")

if __name__ == "__main__":
    # 샘플 데이터 생성
    sample_df = create_data_sample(sample_size=1000)
    
    # 데이터 정보 파일 생성
    create_data_info_file(sample_df)
    
    print("\n✅ 데이터 샘플 생성 완료!")
    print("이제 GitHub에 업로드할 수 있습니다.") 