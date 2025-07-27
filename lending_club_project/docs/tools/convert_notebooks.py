"""
Jupyter Notebook을 Python 스크립트로 변환하는 유틸리티
GitHub 업로드 시 .ipynb 파일 대신 .py 파일 사용을 권장
"""

import nbformat
import os
import re
from pathlib import Path

def convert_notebook_to_script(notebook_path, output_path=None):
    """
    Jupyter Notebook을 Python 스크립트로 변환
    
    Parameters:
    -----------
    notebook_path : str
        입력 .ipynb 파일 경로
    output_path : str, optional
        출력 .py 파일 경로 (None이면 자동 생성)
    """
    
    # 출력 경로가 없으면 자동 생성
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '.py')
    
    # Notebook 로드
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Python 스크립트로 변환
    python_code = []
    
    # 헤더 추가
    python_code.append('"""')
    python_code.append(f'Converted from: {os.path.basename(notebook_path)}')
    python_code.append('Original Jupyter Notebook을 Python 스크립트로 변환')
    python_code.append('"""')
    python_code.append('')
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # 코드 셀
            source = cell.source.strip()
            if source:
                python_code.append('# Code Cell')
                python_code.append(source)
                python_code.append('')
        elif cell.cell_type == 'markdown':
            # 마크다운 셀을 주석으로 변환
            source = cell.source.strip()
            if source:
                python_code.append('# Markdown Cell')
                for line in source.split('\n'):
                    if line.strip():
                        python_code.append(f'# {line}')
                python_code.append('')
    
    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(python_code))
    
    print(f"✅ 변환 완료: {notebook_path} → {output_path}")
    return output_path

def clean_notebook_output(notebook_path, output_path=None):
    """
    Jupyter Notebook의 출력을 제거하여 크기 줄이기
    
    Parameters:
    -----------
    notebook_path : str
        입력 .ipynb 파일 경로
    output_path : str, optional
        출력 .ipynb 파일 경로
    """
    
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '_clean.ipynb')
    
    # Notebook 로드
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # 출력 제거
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # 실행 결과 제거
            if 'outputs' in cell:
                cell.outputs = []
            # 실행 카운트 제거
            if 'execution_count' in cell:
                cell.execution_count = None
    
    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"✅ 출력 제거 완료: {notebook_path} → {output_path}")
    return output_path

def create_notebook_template(template_name='analysis_template.ipynb'):
    """
    깔끔한 Jupyter Notebook 템플릿 생성
    
    Parameters:
    -----------
    template_name : str
        템플릿 파일명
    """
    
    # 기본 템플릿 구조
    template = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Lending Club 신용평가 모델링 분석\n",
                    "\n",
                    "## 개요\n",
                    "이 노트북은 Lending Club 데이터를 활용한 신용평가 모델링 분석을 수행합니다.\n",
                    "\n",
                    "## 주의사항\n",
                    "- 실행 전 필요한 패키지 설치 확인\n",
                    "- 데이터 파일 경로 확인\n",
                    "- 랜덤 시드 설정으로 재현성 보장"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 필요한 라이브러리 임포트\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.metrics import classification_report\n",
                    "\n",
                    "# 재현성을 위한 랜덤 시드 설정\n",
                    "np.random.seed(42)\n",
                    "\n",
                    "# 시각화 설정\n",
                    "plt.style.use('seaborn-v0_8')\n",
                    "sns.set_palette(\"husl\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. 데이터 로드 및 탐색"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 데이터 로드\n",
                    "# df = pd.read_csv('lending_club_2020_train.csv')\n",
                    "# print(f\"데이터 크기: {df.shape}\")\n",
                    "# print(f\"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # 템플릿 저장
    with open(template_name, 'w', encoding='utf-8') as f:
        nbformat.write(template, f)
    
    print(f"✅ 템플릿 생성 완료: {template_name}")

if __name__ == "__main__":
    print("Jupyter Notebook 변환 도구")
    print("=" * 40)
    
    # 현재 디렉토리의 모든 .ipynb 파일 찾기
    notebook_files = list(Path('.').glob('*.ipynb'))
    
    if notebook_files:
        print(f"발견된 .ipynb 파일: {len(notebook_files)}개")
        for notebook in notebook_files:
            print(f"- {notebook}")
        
        print("\n변환 옵션:")
        print("1. Python 스크립트로 변환")
        print("2. 출력 제거하여 크기 줄이기")
        print("3. 템플릿 생성")
        
        choice = input("\n선택하세요 (1-3): ").strip()
        
        if choice == '1':
            for notebook in notebook_files:
                convert_notebook_to_script(str(notebook))
        elif choice == '2':
            for notebook in notebook_files:
                clean_notebook_output(str(notebook))
        elif choice == '3':
            create_notebook_template()
        else:
            print("잘못된 선택입니다.")
    else:
        print("현재 디렉토리에 .ipynb 파일이 없습니다.")
        print("템플릿을 생성합니다...")
        create_notebook_template() 