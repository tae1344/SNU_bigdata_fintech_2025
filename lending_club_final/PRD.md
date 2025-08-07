## Lending Club 신용평가 프로젝트 기획서

### 1. 프로젝트 개요

- **프로젝트명**: Lending Club 부도예측 및 투자포트폴리오 최적화
- **기간**: 2025년 7월 31일 ~ 2025년 8월 8일(보고서 제출 및 발표 일정 별도 공지)
- **목표**: P2P 대출 데이터 기반으로 부도 확률을 예측하고 Sharpe Ratio를 극대화하는 투자 의사결정 모델 설계·구현·평가

### 2. 주요 성과지표

1. **Sharpe Ratio**: 반복 실험(100~1,000회) 결과의 평균 및 분포
2. **IRR(Internal Rate of Return)**: EMI 방식을 적용한 개별 대출 건별 수익률
3. **모델 ROC-AUC & KS 지표**: 예측력 보조 지표
4. **포트폴리오 성능 비교**: 전수 대출 vs. 모델 기반 승인 포트폴리오

### 3. 데이터 설명 및 준비

- **데이터셋**
  - 총 292만 건, 141개 변수 (2007~2020)
  - 훈련용(약 175만 건), 테스트용(약 117만 건, 최종 발표 1주일 전 공개)
  - 변수 설명 문서 숙지

#### 3.1 변수 분류 및 활용 전략

**1. 인구통계 및 개인속성** (Borrower Demographics & Profile)

- emp_title
- emp_length
- home_ownership
- annual_inc
- verification_status
- addr_state
- zip_code

**2. 신용 이력 지표** (Credit History Metrics)

- fico_range_low
- fico_range_high
- earliest_cr_line
- delinq_2yrs
- inq_last_6mths
- pub_rec
- revol_bal
- revol_util
- total_acc
- mths_since_last_record
- mths_since_recent_bc_dlq
- mths_since_last_major_derog
- mths_since_recent_revol_delinq
- mths_since_last_delinq
- mths_since_rcnt_il

**3. 계정 활동 지표** (Account Activity Metrics)

- open_acc
- open_acc_6m
- open_act_il
- open_il_12m
- open_il_24m
- open_rv_12m
- open_rv_24m
- total_cu_tl
- num_tl_30dpd
- num_tl_120dpd_2m

**4. 부채 비율 및 상환 지표** (Debt Ratio & Repayment Metrics)

- dti
- total_bal_ex_mort
- il_util
- all_util
- bc_util
- percent_bc_gt_75
- tot_bal_il
- max_bal_bc
- bc_open_to_buy
- bc_util
- mths_since_recent_bc

**5. 대출 조건 및 실행 정보** (Loan Terms & Origination Details)

- id
- loan_amnt
- funded_amnt
- funded_amnt_inv
- term
- int_rate
- installment
- grade
- sub_grade
- issue_d
- purpose
- pymnt_plan
- title
- initial_list_status
- url

**6. 하드십 플랜 지표** (Hardship Plan Indicators)

- hardship_flag
- hardship_type
- hardship_reason
- hardship_status
- hardship_start_date
- hardship_end_date
- payment_plan_start_date
- hardship_length
- deferral_term
- hardship_dpd
- hardship_loan_status
- hardship_amount
- orig_projected_additional_accrued_interest
- hardship_payoff_balance_amount
- hardship_last_payment_amount

**7. 세컨더리 신청인 정보** (Co-borrower / Joint Application Metrics)

- application_type
- annual_inc_joint
- dti_joint
- verification_status_joint
- revol_bal_joint
- sec_app_earliest_cr_line
- sec_app_inq_last_6mths
- sec_app_mort_acc
- sec_app_open_acc
- sec_app_revol_util
- sec_app_open_act_il
- sec_app_num_rev_accts
- sec_app_chargeoff_within_12_mths
- sec_app_collections_12_mths_ex_med

**8. 모니터링 및 결과 변수** (Monitoring & Outcome Variables)

- loan_status
- last_pymnt_d
- last_pymnt_amnt
- total_pymnt
- total_pymnt_inv
- total_rec_prncp
- total_rec_int
- total_rec_late_fee
- recoveries
- collection_recovery_fee
- out_prncp
- out_prncp_inv

**9. 사용하지 않는 변수 (Excludable / Dropped Variables)**

- **사후 정보(Post-Origination / Leakage Variables)**: 모델 학습 시 부도 예측에 직접 활용되지 않고, 실제 승인 이후에 발생하는 정보를 포함
  - last_pymnt_d
  - last_pymnt_amnt
  - total_pymnt
  - total_pymnt_inv
  - total_rec_prncp
  - total_rec_int
  - total_rec_late_fee
  - recoveries
  - collection_recovery_fee
  - out_prncp
  - out_prncp_inv
  - next_pymnt_d
  - last_credit_pull_d
  - last_fico_range_high
  - last_fico_range_low
- **정책/메타데이터(모델 목표와 무관)**:
  - url
  - policy_code
  - collection_recovery_fee (중복성)
  - application_type (분석 대상은 개별 대출, 공동 대출은 Joint 별도 처리)
- **희귀/중복 변수**:
  - tax_liens
  - pub_rec_bankruptcies
  - chargeoff_within_12_mths
  - sec_app_chargeoff_within_12_mths
  - mths_since_recent_bc_dlq, mths_since_recent_revol_delinq (신용 연체 변수 중 복수)
  - orig_projected_additional_accrued_interest (하드십 파생 변수, null 다수)

#### 3.2 EDA 및 데이터 전처리

1. **초기 데이터 확인(Profiling)**

   - 샘플링
     ```python
     # pandas를 사용한 샘플 추출
     df_sample = df.sample(frac=0.02, random_state=42)
     ```
   - 데이터 타입 점검
     ```python
     df_sample.dtypes.value_counts()
     ```
   - 기초 통계량 확인
     ```python
     df_sample.describe(include='all')
     ```

2. **결측치 처리**

   - 결측치 시각화
     ```python
     import missingno as msno
     msno.matrix(df_sample)
     ```
   - 패턴 분석 및 처리 예시
     ```python
     # 중요 변수 중앙값 대체
     df['annual_inc'].fillna(df['annual_inc'].median(), inplace=True)
     # 하드십 변수 null 별도 범주
     df['hardship_type'].fillna('None', inplace=True)
     ```

3. **이상치 탐지 및 처리**

   - IQR 기반 이상치 식별
     ```python
     Q1 = df['loan_amnt'].quantile(0.25)
     Q3 = df['loan_amnt'].quantile(0.75)
     IQR = Q3 - Q1
     df_out = df[(df['loan_amnt'] >= Q1 - 1.5 * IQR) & (df['loan_amnt'] <= Q3 + 1.5 * IQR)]
     ```
   - 로그 변환 예시
     ```python
     df['revol_bal_log'] = np.log1p(df['revol_bal'])
     ```

4. **변수 변환 및 스케일링**

   - 표준화
     ```python
     from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     df[['dti', 'revol_util']] = scaler.fit_transform(df[['dti', 'revol_util']])
     ```
   - 범주형 인코딩
     ```python
     df = pd.get_dummies(df, columns=['grade', 'home_ownership'], drop_first=True)
     ```
   - 날짜 파생 변수
     ```python
     df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
     df['issue_year'] = df['issue_d'].dt.year
     df['issue_month'] = df['issue_d'].dt.month
     ```

5. **피처 엔지니어링**

   - 부채 대비 소득 비율
     ```python
     df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
     ```
   - 상호작용 변수
     ```python
     df['rate_term_interaction'] = df['int_rate'] * df['term']
     ```

6. **데이터 분할 및 교차검증 설정**

   ```python
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   for train_idx, val_idx in skf.split(df, df['loan_status']):
       df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
   ```

7. **EDA 결과 시각화**

   - 부도 그룹 분포 비교
     ```python
     import seaborn as sns
     sns.boxplot(x='loan_status', y='annual_inc', data=df)
     ```
   - 상관관계 히트맵
     ```python
     corr = df.corr()
     sns.heatmap(corr, cmap='coolwarm', annot=False)
     ```

### 4. 분석·모델링 전략

1. **EDA & 변수 중요도 탐색**

   - **Univariate Analysis**
     - 수치형 변수 분포 확인: 히스토그램, KDE, 박스플롯
     - 예시 코드:
       ```python
       import matplotlib.pyplot as plt
       df['fico_range_low'].hist(bins=30)
       plt.title('FICO Range Low Distribution')
       plt.show()
       ```
     - 범주형 변수 분포 비교: 막대그래프, 파이차트
       ```python
       df['grade'].value_counts().plot(kind='bar')
       ```
   - **Bivariate Analysis**
     - 타깃(loan_status) 대비 변수 분포: 그룹별 통계, 박스플롯
       ```python
       import seaborn as sns
       sns.boxplot(x='loan_status', y='annual_inc', data=df)
       ```
     - 범주형 vs. 타깃: 교차표와 부도율 계산
       ```python
       pd.crosstab(df['grade'], df['loan_status'], normalize='index')
       ```
   - **상관관계 분석 (Correlation Analysis)**
     - 피처 간 상관관계 행렬 및 히트맵
       ```python
       corr = df.corr()
       sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
       ```
     - 타깃과의 상관도: 상위 10개 상관변수 선별
       ```python
       corr_with_target = corr['loan_status'].abs().sort_values(ascending=False)
       top_features = corr_with_target[1:11]
       ```
   - **Feature Importance 탐색**
     - 트리 기반 모델(랜덤포레스트)로 피처 중요도 추출
       ```python
       from sklearn.ensemble import RandomForestClassifier
       model = RandomForestClassifier(n_estimators=100, random_state=42)
       model.fit(X_train, y_train)
       importances = pd.Series(model.feature_importances_, index=X_train.columns)
       importances.nlargest(10).plot(kind='barh')
       ```
     - 상호 정보량(Mutual Information) 기반 선택
       ```python
       from sklearn.feature_selection import mutual_info_classif
       mi = mutual_info_classif(X_train, y_train)
       mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
       mi_series.head(10)
       ```
   - **차원 축소 및 시각화**
     - PCA로 주요 컴포넌트 파악 및 분산 설명 비율
       ```python
       from sklearn.decomposition import PCA
       pca = PCA(n_components=5)
       pca.fit(X_scaled)
       print(pca.explained_variance_ratio_)
       ```
     - 2D 투영 결과 시각화
       ```python
       X_pca = pca.transform(X_scaled)
       plt.scatter(X_pca[:,0], X_pca[:,1], c=y, alpha=0.3)
       plt.xlabel('PC1'); plt.ylabel('PC2')
       ```
   - **결과 종합 및 변수 선정**
     - 상관, 중요도, MI 결과를 종합하여 유의 피처 리스트 생성
     - L1 정규화 로지스틱 회귀로 추가 검증
       ```python
       from sklearn.linear_model import LogisticRegression
       l1_model = LogisticRegression(penalty='l1', solver='liblinear')
       l1_model.fit(X_train, y_train)
       coef_series = pd.Series(abs(l1_model.coef_[0]), index=X_train.columns)
       coef_series[coef_series>0].sort_values(ascending=False)
       ```

**2. 모델 후보군**

- 로지스틱 회귀
- 랜덤 포레스트 / XGBoost / LightGBM
- 신경망(MLP)
- 앙상블(스태킹, 배깅)

3. **하이퍼파라미터 튜닝**

   - **Grid Search**
     ```python
     from sklearn.model_selection import GridSearchCV
     param_grid = {
         'n_estimators': [100, 200, 500],
         'max_depth': [3, 5, 10],
         'min_samples_split': [2, 5, 10]
     }
     grid_search = GridSearchCV(
         estimator=RandomForestClassifier(random_state=42),
         param_grid=param_grid,
         scoring='roc_auc',
         cv=5,
         n_jobs=-1
     )
     grid_search.fit(X_train, y_train)
     print(grid_search.best_params_)
     ```
   - **Random Search**
     ```python
     from sklearn.model_selection import RandomizedSearchCV
     param_dist = {
         'n_estimators': [100, 200, 500, 1000],
         'max_depth': [None, 5, 10, 20],
         'min_samples_split': [2, 5, 10, 20],
         'max_features': ['auto', 'sqrt', 'log2']
     }
     random_search = RandomizedSearchCV(
         estimator=XGBClassifier(random_state=42),
         param_distributions=param_dist,
         n_iter=50,
         scoring='roc_auc',
         cv=5,
         n_jobs=-1,
         random_state=42
     )
     random_search.fit(X_train, y_train)
     print(random_search.best_params_)
     ```
   - **Bayesian Optimization**
     ```python
     from skopt import BayesSearchCV
     bayes_search = BayesSearchCV(
         estimator=LightGBMClassifier(random_state=42),
         search_spaces={
             'num_leaves': (20, 150),
             'learning_rate': (1e-3, 1e-1, 'log-uniform'),
             'n_estimators': (50, 500)
         },
         n_iter=30,
         scoring='roc_auc',
         cv=5,
         n_jobs=-1,
         random_state=42
     )
     bayes_search.fit(X_train, y_train)
     print(bayes_search.best_params_)
     ```
   - **Threshold Optimization**
     - Validation set에서 예측 확률 기반 최적 threshold 탐색
     ```python
     from sklearn.metrics import precision_recall_curve
     y_scores = model.predict_proba(X_val)[:,1]
     precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
     f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
     best_idx = f1_scores.argmax()
     best_threshold = thresholds[best_idx]
     print('Best threshold:', best_threshold)
     ```

4. **IRR 계산 모듈**

   - EMI 월별 현금흐름 시뮬레이터 구현
   - IRR 산출 및 검증

5. **Sharpe Ratio 최적화 로직**

   - 예측 결과 기반 포트폴리오 시뮬레이션
   - 국채 수익률 반영(3년/5년) 계산기 통합

### 5. 실험 설계 및 검증

- **반복 실험**: k-fold 또는 Monte Carlo CV 방식으로 100~1,000회 분할
- **결과 저장·시각화**
  - Sharpe Ratio 분포 히스토그램
  - 수익률 vs. 위험 기여도 분석 차트
  - ROC, PR 곡선
- **벤치마크**: 모든 대출 승인 시 Sharpe Ratio

### 6. 인프라 및 도구

- **개발 환경**: Python, Jupyter Notebook / VSCode
- **라이브러리**: pandas, numpy, scikit-learn, xgboost, lightgbm, statsmodels
- **버전 관리**: GitHub (협업 브랜치 전략)

### 8. 리스크 관리

- **데이터 크기**: 샘플링 전략 및 병렬 처리 스크립트 마련
- **모델 과적합**: CV & 검증 절차 철저 적용
- **시간 부족**: 핵심 모듈 우선 개발, 자동화 스크립트 활용
- **협업 지연**: 매일 스탠드업 미팅으로 진행 상황 점검
