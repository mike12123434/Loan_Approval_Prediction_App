"""
貸款風險評估系統 - 配置文件
集中管理所有常數、超參數和閾值
"""

import os

# ==================== 文件路徑 ====================
DATA_FILE = 'loan_approval_dataset.csv'
MODEL_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'loan_classifier.pkl')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'rate_regressor.pkl')
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

# ==================== 模型參數 ====================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15  # 從訓練集中再分出驗證集用於調參

# ==================== 進階 ML 配置 ====================

# 特徵選擇參數
FEATURE_SELECTION_ENABLED = True
CORRELATION_THRESHOLD = 0.95  # 移除高度相關特徵
FEATURE_IMPORTANCE_THRESHOLD = 0.01  # 移除重要性過低特徵
P_VALUE_THRESHOLD = 0.05  # 統計顯著性閾值

# 交叉驗證參數
CV_FOLDS = 5
CV_SCORING = 'roc_auc'  # 評估指標

# 超參數搜尋空間 (Grid Search / Random Search)
HYPERPARAMETER_TUNING_ENABLED = True
TUNING_METHOD = 'random'  # 'grid' or 'random'
RANDOM_SEARCH_ITERATIONS = 50

# ==================== 模型參數 ====================

# XGBoost 分類器參數 (Gradient Boosting - 優於 Random Forest)
XGBOOST_CLASSIFIER_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,  # 正則化參數
    'reg_alpha': 0.1,  # L1 正則化
    'reg_lambda': 1.0,  # L2 正則化
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'scale_pos_weight': 1,  # 處理類別不平衡
    'eval_metric': 'logloss'
}

# XGBoost 分類器超參數搜尋空間
XGBOOST_CLASSIFIER_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0]
}

# Random Forest 分類器參數 (作為 baseline)
RANDOM_FOREST_CLASSIFIER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',  # 降維策略 (blessing of non-uniformity)
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'max_samples': 0.8  # Bootstrap 抽樣比例
}

# XGBoost 迴歸器參數
XGBOOST_REGRESSOR_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,  # L1 正則化
    'reg_lambda': 1.0,  # L2 正則化
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# 模型選擇
USE_XGBOOST = True  # True: XGBoost, False: Random Forest
USE_ENSEMBLE = True  # 使用多模型集成

# Stacking 集成參數
STACKING_ENABLED = True
STACKING_CV_FOLDS = 3

# ==================== 信用評分參數 ====================
CREDIT_SCORE_BASE = 650
CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 900

# 收入等級與加分
INCOME_TIERS = [
    (100000, 100),  # 月收入 >= 10萬，加100分
    (60000, 50),    # 月收入 >= 6萬，加50分
    (40000, 20),    # 月收入 >= 4萬，加20分
]

# 負債收入比與加減分
DTI_TIERS = [
    (0.2, 100),   # DTI < 20%，加100分
    (0.4, 50),    # DTI < 40%，加50分
    (0.7, -100),  # DTI > 70%，減100分
]

EMPLOYMENT_BONUS = 30  # 受薪員工加分
DEPENDENT_PENALTY = 15  # 每位扶養人扣分

# ==================== 利率計算公式參數 ====================
# 利率 = BASE_RATE + (信用評分影響) + (負債比影響) + (貸款金額影響)

INTEREST_RATE_BASE = 3.0  # 基礎利率
INTEREST_RATE_MIN = 2.5
INTEREST_RATE_MAX = 12.0

# 信用評分對利率的影響（分數越高，利率越低）
# 利率增幅 = (900 - 信用評分) / CREDIT_SCORE_FACTOR
CREDIT_SCORE_FACTOR = 80

# 負債收入比對利率的影響
# DTI 每增加 0.1，利率增加 DTI_RATE_MULTIPLIER
DTI_RATE_MULTIPLIER = 3.0

# 貸款金額對收入比的影響
# 貸款/年收入 每增加 1，利率增加 LOAN_TO_INCOME_MULTIPLIER
LOAN_TO_INCOME_MULTIPLIER = 0.3

# ==================== 評估閾值（將通過數據驗證設定）====================
# 這些是初始值，訓練後會根據 ROC 曲線和業務需求調整

# 核准機率閾值（預設 0.5，訓練後會優化）
APPROVAL_THRESHOLD = 0.5

# 信用評分閾值
CREDIT_SCORE_EXCELLENT = 750  # 優質客戶
CREDIT_SCORE_GOOD = 650       # 良好
CREDIT_SCORE_POOR = 500       # 不良

# 負債收入比閾值
DTI_EXCELLENT = 0.2   # 優質（低風險）
DTI_ACCEPTABLE = 0.4  # 可接受
DTI_HIGH_RISK = 0.7   # 高風險

# 貸款金額與收入比閾值
LOAN_TO_INCOME_MAX = 5.0  # 最高為年收入的5倍

# ==================== 資產估算參數 ====================
# 基於收入的資產估算係數
ASSET_RESIDENTIAL_MULTIPLIER = 2.0    # 住宅資產 = 年收入 × 2
ASSET_COMMERCIAL_EMPLOYED = 0.3       # 受薪者商業資產
ASSET_COMMERCIAL_SELFEMPLOYED = 0.5   # 自僱者商業資產
ASSET_LUXURY_MULTIPLIER = 0.3         # 奢侈品資產
ASSET_BANK_MULTIPLIER = 1.0           # 銀行存款

# 信用卡使用率估算
CREDIT_CARD_USAGE_RATE = 0.05  # 假設使用額度的5%

# ==================== 必要欄位 ====================
REQUIRED_COLUMNS = [
    'loan_status', 'income_annum', 'loan_amount', 'cibil_score',
    'education', 'self_employed', 'residential_assets_value',
    'commercial_assets_value', 'luxury_assets_value', 
    'bank_asset_value', 'loan_term', 'no_of_dependents'
]

# 特徵欄位
FEATURE_COLUMNS = [
    'no_of_dependents', 'education_encoded', 'self_employed_encoded',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value', 'total_assets',
    'loan_to_income_ratio', 'asset_to_loan_ratio'
]

# 類別欄位的所有可能值（避免 LabelEncoder 失敗）
EDUCATION_VALUES = [' Graduate', ' Not Graduate']
EMPLOYMENT_VALUES = [' No', ' Yes']

# ==================== UI 配置 ====================
PAGE_TITLE = "信貸試算系統"
PAGE_ICON = "💰"
MAX_WIDTH = 800

# 輸入欄位的預設值和範圍
INCOME_MIN = 10000
INCOME_MAX = 500000
INCOME_DEFAULT = 50000

LOAN_MIN = 50000
LOAN_MAX = 10000000
LOAN_DEFAULT = 500000

TERM_OPTIONS = [3, 5, 7, 10]
TERM_DEFAULT = 5
