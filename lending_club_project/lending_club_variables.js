/*
  1. 카테고리별 분류
    17개 카테고리로 변수들을 논리적으로 그룹화
    각 카테고리별 변수 개수도 포함
  2. 주요 카테고리
    대출자 정보(borrower_information) (11개): 기본 인적사항, 소득, 고용 정보
    대출 정보(loan_information) (15개): 대출 금액, 이자율, 등급, 목적 등
    신용 정보(credit_information) (8개): FICO 점수, DTI 비율, 신용 이력
    지급 정보(payment_information) (11개): 상환 관련 모든 정보
    연체 정보(delinquency_information) (12개): 과거 연체 이력 및 현재 상태
    계좌 정보(account_information) (20개): 신용 계좌 수, 개설 이력 등
    잔액 정보(balance_information) (15개): 신용 한도, 사용률 등
    공동신청인 정보(secondary_applicant_information) (13개): 공동 대출 시 추가 정보
    어려움 대출 정보(hardship_information) (15개): 어려움 대출 관련 모든 정보
    부채 조정 정보(debt_settlement_information) (7개): 부채 조정 관련 정보
    메타데이터(metadata) (1개): 대출 정보 링크
  3. 프로젝트 활용 방안
    이 JSON 파일은 다음과 같이 활용할 수 있습니다:
    특성 엔지니어링: 관련 변수들을 그룹화하여 새로운 특성 생성
    모델링 전략: 카테고리별로 다른 전처리 방법 적용
    변수 선택: 각 카테고리에서 중요한 변수 선별
    결과 해석: 모델 성능을 카테고리별로 분석
*/

const lending_club_variables_json = {
  lending_club_variables: {
    borrower_information: {
      id: "A unique LC assigned ID for the loan listing",
      member_id: "A unique LC assigned Id for the borrower member",
      annual_inc:
        "The self-reported annual income provided by the borrower during registration",
      annual_inc_joint:
        "The combined self-reported annual income provided by the co-borrowers during registration",
      application_type:
        "Indicates whether the loan is an individual application or a joint application with two co-borrowers",
      emp_length:
        "Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years",
      emp_title:
        "The job title supplied by the Borrower when applying for the loan",
      home_ownership:
        "The home ownership status provided by the borrower during registration or obtained from the credit report. Values are: RENT, OWN, MORTGAGE, OTHER",
      addr_state: "The state provided by the borrower in the loan application",
      zip_code:
        "The first 3 numbers of the zip code provided by the borrower in the loan application",
      verification_status:
        "Indicates if income was verified by LC, not verified, or if the income source was verified",
      verified_status_joint:
        "Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified",
    },
    loan_information: {
      loan_amnt:
        "The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value",
      funded_amnt:
        "The total amount committed to that loan at that point in time",
      funded_amnt_inv:
        "The total amount committed by investors for that loan at that point in time",
      term: "The number of payments on the loan. Values are in months and can be either 36 or 60",
      int_rate: "Interest Rate on the loan",
      installment:
        "The monthly payment owed by the borrower if the loan originates",
      grade: "LC assigned loan grade",
      sub_grade: "LC assigned loan subgrade",
      loan_status: "Current status of the loan",
      purpose: "A category provided by the borrower for the loan request",
      title: "The loan title provided by the borrower",
      desc: "Loan description provided by the borrower",
      issue_d: "The month which the loan was funded",
      initial_list_status:
        "The initial listing status of the loan. Possible values are – W, F",
      policy_code:
        "publicly available policy_code=1, new products not publicly available policy_code=2",
      disbursement_method:
        "The method by which the borrower receives their loan. Possible values are: CASH, DIRECT_PAY",
    },
    credit_information: {
      fico_range_low:
        "The lower boundary range the borrower's FICO at loan origination belongs to",
      fico_range_high:
        "The upper boundary range the borrower's FICO at loan origination belongs to",
      last_fico_range_low:
        "The lower boundary range the borrower's last FICO pulled belongs to",
      last_fico_range_high:
        "The upper boundary range the borrower's last FICO pulled belongs to",
      earliest_cr_line:
        "The month the borrower's earliest reported credit line was opened",
      last_credit_pull_d:
        "The most recent month LC pulled credit for this loan",
      dti: "A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income",
      dti_joint:
        "A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income",
    },
    payment_information: {
      last_pymnt_amnt: "Last total payment amount received",
      last_pymnt_d: "Last month payment was received",
      next_pymnt_d: "Next scheduled payment date",
      total_pymnt: "Payments received to date for total amount funded",
      total_pymnt_inv:
        "Payments received to date for portion of total amount funded by investors",
      total_rec_prncp: "Principal received to date",
      total_rec_int: "Interest received to date",
      total_rec_late_fee: "Late fees received to date",
      out_prncp: "Remaining outstanding principal for total amount funded",
      out_prncp_inv:
        "Remaining outstanding principal for portion of total amount funded by investors",
      pymnt_plan:
        "Indicates if a payment plan has been put in place for the loan",
    },
    delinquency_information: {
      acc_now_delinq:
        "The number of accounts on which the borrower is now delinquent",
      delinq_2yrs:
        "The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years",
      delinq_amnt:
        "The past-due amount owed for the accounts on which the borrower is now delinquent",
      mths_since_last_delinq:
        "The number of months since the borrower's last delinquency",
      mths_since_last_major_derog:
        "Months since most recent 90-day or worse rating",
      mths_since_last_record:
        "The number of months since the last public record",
      num_tl_30dpd:
        "Number of accounts currently 30 days past due (updated in past 2 months)",
      num_tl_120dpd_2m:
        "Number of accounts currently 120 days past due (updated in past 2 months)",
      num_tl_90g_dpd_24m:
        "Number of accounts 90 or more days past due in last 24 months",
      num_accts_ever_120_pd:
        "Number of accounts ever 120 or more days past due",
      mths_since_recent_revol_delinq:
        "Months since most recent revolving delinquency",
      mths_since_recent_bc_dlq: "Months since most recent bankcard delinquency",
    },
    account_information: {
      total_acc:
        "The total number of credit lines currently in the borrower's credit file",
      open_acc: "The number of open credit lines in the borrower's credit file",
      open_acc_6m: "Number of open trades in last 6 months",
      open_il_12m: "Number of installment accounts opened in past 12 months",
      open_il_24m: "Number of installment accounts opened in past 24 months",
      open_rv_12m: "Number of revolving trades opened in past 12 months",
      open_rv_24m: "Number of revolving trades opened in past 24 months",
      open_act_il: "Number of currently active installment trades",
      open_act_rev_tl: "Number of currently active revolving trades",
      num_actv_bc_tl: "Number of currently active bankcard accounts",
      num_actv_rev_tl: "Number of currently active revolving trades",
      num_bc_sats: "Number of satisfactory bankcard accounts",
      num_bc_tl: "Number of bankcard accounts",
      num_il_tl: "Number of installment accounts",
      num_op_rev_tl: "Number of open revolving accounts",
      num_rev_accts: "Number of revolving accounts",
      num_rev_tl_bal_gt_0: "Number of revolving trades with balance >0",
      num_sats: "Number of satisfactory accounts",
      num_tl_op_past_12m: "Number of accounts opened in past 12 months",
      acc_open_past_24mths: "Number of trades opened in past 24 months",
    },
    balance_information: {
      revol_bal: "Total credit revolving balance",
      revol_util:
        "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit",
      all_util: "Balance to credit limit on all trades",
      bc_util:
        "Ratio of total current balance to high credit/credit limit for all bankcard accounts",
      il_util:
        "Ratio of total current balance to high credit/credit limit on all install acct",
      avg_cur_bal: "Average current balance of all accounts",
      bc_open_to_buy: "Total open to buy on revolving bankcards",
      max_bal_bc: "Maximum current balance owed on all revolving accounts",
      tot_cur_bal: "Total current balance of all accounts",
      tot_hi_cred_lim: "Total high credit/credit limit",
      total_bal_ex_mort: "Total credit balance excluding mortgage",
      total_bal_il: "Total current balance of all installment accounts",
      total_bc_limit: "Total bankcard high credit/credit limit",
      total_il_high_credit_limit: "Total installment high credit/credit limit",
      total_rev_hi_lim: "Total revolving high credit/credit limit",
    },
    inquiry_information: {
      inq_fi: "Number of personal finance inquiries",
      inq_last_12m: "Number of credit inquiries in past 12 months",
      inq_last_6mths:
        "The number of inquiries in past 6 months (excluding auto and mortgage inquiries)",
      mths_since_recent_inq: "Months since most recent inquiry",
    },
    account_age_information: {
      mo_sin_old_il_acct: "Months since oldest bank installment account opened",
      mo_sin_old_rev_tl_op: "Months since oldest revolving account opened",
      mo_sin_rcnt_rev_tl_op:
        "Months since most recent revolving account opened",
      mo_sin_rcnt_tl: "Months since most recent account opened",
      mths_since_rcnt_il:
        "Months since most recent installment accounts opened",
      mths_since_recent_bc: "Months since most recent bankcard account opened",
    },
    mortgage_information: {
      mort_acc: "Number of mortgage accounts",
    },
    public_records: {
      pub_rec: "Number of derogatory public records",
      pub_rec_bankruptcies: "Number of public record bankruptcies",
      tax_liens: "Number of tax liens",
    },
    collections_and_chargeoffs: {
      chargeoff_within_12_mths: "Number of charge-offs within 12 months",
      collections_12_mths_ex_med:
        "Number of collections in 12 months excluding medical collections",
      collection_recovery_fee: "post charge off collection fee",
      recoveries: "post charge off gross recovery",
      tot_coll_amt: "Total collection amounts ever owed",
    },
    credit_quality_metrics: {
      pct_tl_nvr_dlq: "Percent of trades never delinquent",
      percent_bc_gt_75: "Percentage of all bankcard accounts > 75% of limit",
      total_cu_tl: "Number of finance trades",
    },
    secondary_applicant_information: {
      sec_app_fico_range_low: "FICO range (high) for the secondary applicant",
      sec_app_fico_range_high: "FICO range (low) for the secondary applicant",
      sec_app_earliest_cr_line:
        "Earliest credit line at time of application for the secondary applicant",
      sec_app_inq_last_6mths:
        "Credit inquiries in the last 6 months at time of application for the secondary applicant",
      sec_app_mort_acc:
        "Number of mortgage accounts at time of application for the secondary applicant",
      sec_app_open_acc:
        "Number of open trades at time of application for the secondary applicant",
      sec_app_revol_util:
        "Ratio of total current balance to high credit/credit limit for all revolving accounts",
      sec_app_open_act_il:
        "Number of currently active installment trades at time of application for the secondary applicant",
      sec_app_num_rev_accts:
        "Number of revolving accounts at time of application for the secondary applicant",
      sec_app_chargeoff_within_12_mths:
        "Number of charge-offs within last 12 months at time of application for the secondary applicant",
      sec_app_collections_12_mths_ex_med:
        "Number of collections within last 12 months excluding medical collections at time of application for the secondary applicant",
      sec_app_mths_since_last_major_derog:
        "Months since most recent 90-day or worse rating at time of application for the secondary applicant",
      revol_bal_joint:
        "Sum of revolving credit balance of the co-borrowers, net of duplicate balances",
    },
    hardship_information: {
      hardship_flag: "Flags whether or not the borrower is on a hardship plan",
      hardship_type: "Describes the hardship plan offering",
      hardship_reason: "Describes the reason the hardship plan was offered",
      hardship_status:
        "Describes if the hardship plan is active, pending, canceled, completed, or broken",
      deferral_term:
        "Amount of months that the borrower is expected to pay less than the contractual monthly payment amount due to a hardship plan",
      hardship_amount:
        "The interest payment that the borrower has committed to make each month while they are on a hardship plan",
      hardship_start_date: "The start date of the hardship plan period",
      hardship_end_date: "The end date of the hardship plan period",
      payment_plan_start_date:
        "The day the first hardship plan payment is due. For example, if a borrower has a hardship plan period of 3 months, the start date is the start of the three-month period in which the borrower is allowed to make interest-only payments",
      hardship_length:
        "The number of months the borrower will make smaller payments than normally obligated due to a hardship plan",
      hardship_dpd: "Account days past due as of the hardship plan start date",
      hardship_loan_status: "Loan Status as of the hardship plan start date",
      orig_projected_additional_accrued_interest:
        "The original projected additional interest amount that will accrue for the given hardship payment plan as of the Hardship Start Date. This field will be null if the borrower has broken their hardship payment plan",
      hardship_payoff_balance_amount:
        "The payoff balance amount as of the hardship plan start date",
      hardship_last_payment_amount:
        "The last payment amount as of the hardship plan start date",
    },
    debt_settlement_information: {
      debt_settlement_flag:
        "Flags whether or not the borrower, who has charged-off, is working with a debt-settlement company",
      debt_settlement_flag_date:
        "The most recent date that the Debt_Settlement_Flag has been set",
      settlement_status:
        "The status of the borrower's settlement plan. Possible values are: COMPLETE, ACTIVE, BROKEN, CANCELLED, DENIED, DRAFT",
      settlement_date:
        "The date that the borrower agrees to the settlement plan",
      settlement_amount:
        "The loan amount that the borrower has agreed to settle for",
      settlement_percentage:
        "The settlement amount as a percentage of the payoff balance amount on the loan",
      settlement_term:
        "The number of months that the borrower will be on the settlement plan",
    },
    metadata: {
      url: "URL for the LC page with listing data",
    },
  },
  variable_count: 141,
  categories: {
    borrower_information: 11,
    loan_information: 15,
    credit_information: 8,
    payment_information: 11,
    delinquency_information: 12,
    account_information: 20,
    balance_information: 15,
    inquiry_information: 4,
    account_age_information: 6,
    mortgage_information: 1,
    public_records: 3,
    collections_and_chargeoffs: 5,
    credit_quality_metrics: 3,
    secondary_applicant_information: 13,
    hardship_information: 15,
    debt_settlement_information: 7,
    metadata: 1,
  },
};
