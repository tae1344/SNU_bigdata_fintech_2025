# PRD: 불륜 확률 시각화 (웹) — 데이터셋 변수 반영 버전

## 0. 한줄 요약

사용자가 `rate_marriage`, `age`, `yrs_married`, `children`, `religious`, `educ`, `occupation`, `occupation_husb` 값을 설정하면, 가상의 “페르소나 캐릭터”가 생성되고 추정 **불륜 확률**(엔터테인먼트용 점수)을 게이지와 애니메이션으로 시각화한다.  
확률이 높을수록 캐릭터가 더 빠르고 멀리 도망가는 장면을 보여준다.

---

## 1. 목표 & 비즈니스 목적

- **핵심 목표**: 데이터 기반 입력 → 직관적 시각화 → 재미와 공유성 높은 인터랙션.
- **성과 지표(KPI)**:
  - 평균 세션 길이 ≥ 2분
  - 공유율 ≥ 8%
  - 완주율 ≥ 70%

---

## 2. 범위 (Scope)

### In

- **입력 필터**:
  - `rate_marriage` (결혼 만족도 1~5)
  - `age` (나이)
  - `yrs_married` (결혼 기간, 년 단위)
  - `children` (자녀 수)
  - `religious` (종교 성향 1~4, 높을수록 종교적)
  - `educ` (교육 수준, 코드 값)
  - `occupation` (사용자 직업 코드)
  - `occupation_husb` (배우자 직업 코드)
- 점수 계산 로직: 실제 데이터셋 기반 통계/모델의 예측 결과 또는 로컬 가중치 계산
- 결과 시각화: 게이지 + 영향 요인 TOP3 + 애니메이션
- 비교 모드 및 공유 기능

### Out

- 민감 정보 저장
- 로그인/회원제
- 실제 신원 식별 가능 데이터 처리

---

## 3. 사용자 스토리

- **U1**: “내 조건을 넣으면 불륜 확률 점수와 캐릭터 반응을 보고 싶어요.”
- **U2**: “조건을 바꾸면 즉시 반응이 변하는 걸 보고 싶어요.”
- **U3**: “점수가 왜 이렇게 나왔는지 영향 요인을 알고 싶어요.”

---

## 4. UX 플로우

1. **랜딩**
   - 설명 + 시작 버튼
2. **입력 스텝**
   - Step 1: `rate_marriage`, `age`, `yrs_married`
   - Step 2: `children`, `religious`
   - Step 3: `educ`, `occupation`, `occupation_husb`
3. **결과**
   - 게이지 점수 + 텍스트
   - 영향 요인 TOP3
   - 캐릭터 도망 애니메이션
4. **비교/공유**
   - 기존 시나리오와 변경값 비교
   - 이미지 저장/공유

---

## 5. 기능 요구사항

### 5.1 입력/필터

```ts
type Filters = {
  rate_marriage: 1 | 2 | 3 | 4 | 5; // 결혼 만족도
  age: number; // 나이
  yrs_married: number; // 결혼 기간
  children: number; // 자녀 수
  religious: 1 | 2 | 3 | 4; // 종교 성향
  educ: number; // 교육 수준 코드
  occupation: number; // 직업 코드
  occupation_husb: number; // 배우자 직업 코드
};
```

- 검증: 필수값, 범위 제한(예: age 18~80)

### 5.2 점수 산출

- 모형: 데이터셋 기반 로지스틱 회귀 계수 또는 예측 API
- 계산식 예시

```ini
p = 1 / (1 + e^-(β0 + Σ βi * xi))
score = round(p * 100)
```

- 영향 요인 산출: 각 변수의 βi \* (xi - mean_i) 값을 계산하고, 절대값 기준 상위 3개를 시각화

### 5.3 시각화

- 게이지: 0~100
- 영향 요인 바 차트
- 아바타: age/children/occupation 등에 따라 외형 변화
- 애니메이션: 확률 ↑ → 거리·속도·이펙트 강도 ↑

### 6. 애니메이션 사양

- 도망 정도 매핑:
  - Distance(px) = 80 + 240 \* p
  - Speed(ms) = 2000 - 1000 \* p
  - Effect Level = 1 + floor(2 \* p)

### 7. 아키텍처 & 기술 스택

- FE: Next.js 15, TypeScript, Tailwind, Recoil, Framer Motion, Recharts
- 점수 로직: 로컬 스크립트(scoring.ts) 또는 API 호출

### 8. 데이터모델

```TS
type ScoreResult = {
  score: number; // 0-100
  probability: number; // 0-1
  contributions: { key: keyof Filters; value: number }[];
};

```

### 9. 화면 설계 주요 포인트

- 설문 화면: 직관적 아이콘/슬라이더/드롭다운
- 결과 화면: 게이지 중앙 배치, 아바타 씬 아래 배치
- 비교 모드: 두 카드 나란히

### 10. 분석 이벤트

- start
- filters_complete
- result_view
- share
- compare

### 11. 윤리고지

- 모든 결과는 오락/콘텐츠용이며 실제 사실과 무관
- 민감 정보는 저장하지 않음
- 모든 화면에 고지 문구 표시
