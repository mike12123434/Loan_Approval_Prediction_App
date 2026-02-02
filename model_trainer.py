"""
貸款風險評估系統 - 進階模型訓練模組

整合進階 ML 技術:
=================
1. 表示法 (Representation):
   - XGBoost (Gradient Boosting) with L1/L2 regularization
   - Random Forest (Ensemble of Decision Trees)
   - Stacking Ensemble

2. 評估 (Evaluation):
   - Train/Validation/Test 三集分割
   - K-Fold Cross-Validation
   - 多指標評估 (Accuracy, Precision, Recall, F1, AUC-ROC)
   - ROC 曲線優化閾值

3. 優化 (Optimization):
   - 特徵選擇 (統計顯著性檢定、互信息)
   - 超參數調優 (Random/Grid Search)
   - 正則化 (L1/L2 penalty)
   - Blessing of non-uniformity (特徵重要性加權)
   - 模型可解釋性 (SHAP values)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    RandomizedSearchCV, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings

from config import *
from advanced_ml import FeatureSelector, DimensionalityReducer, ModelExplainer, validate_feature_independence

warnings.filterwarnings('ignore')

# 嘗試導入 XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost 未安裝，將使用 Random Forest")


class AdvancedLoanModelTrainer:
    """進階貸款模型訓練器"""
    
    def __init__(self):
        self.clf = None
        self.reg = None
        self.le_edu = None
        self.le_emp = None
        self.feature_cols = FEATURE_COLUMNS.copy()
        self.selected_features = None
        self.metrics = {}
        self.optimal_threshold = APPROVAL_THRESHOLD
        
        # 進階工具
        self.feature_selector = FeatureSelector(
            correlation_threshold=CORRELATION_THRESHOLD,
            p_value_threshold=P_VALUE_THRESHOLD
        )
        self.dimensionality_reducer = DimensionalityReducer()
        self.explainer = ModelExplainer()
        
    def load_and_prepare_data(self, filepath=DATA_FILE):
        """載入並準備數據"""
        print("\n" + "="*70)
        print("📂 數據載入與準備")
        print("="*70)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到數據檔案: {filepath}")
        
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少欄位: {', '.join(missing_cols)}")
        
        print(f"✅ 原始數據: {len(df)} 筆, {len(df.columns)} 欄")
        
        # 清理數據
        df = df.dropna()
        print(f"✅ 清理後: {len(df)} 筆")
        
        if len(df) == 0:
            raise ValueError("清理後無有效數據")
        
        # 數據分佈分析
        print(f"\n數據分佈:")
        print(f"  核准數量: {(df['loan_status'].str.strip() == 'Approved').sum()} "
              f"({(df['loan_status'].str.strip() == 'Approved').mean()*100:.1f}%)")
        print(f"  拒絕數量: {(df['loan_status'].str.strip() != 'Approved').sum()} "
              f"({(df['loan_status'].str.strip() != 'Approved').mean()*100:.1f}%)")
        
        return df
    
    def feature_engineering(self, df):
        """特徵工程"""
        print("\n" + "="*70)
        print("🔧 特徵工程")
        print("="*70)
        
        df = df.copy()
        
        # 計算衍生特徵
        df['total_assets'] = (
            df['residential_assets_value'] + 
            df['commercial_assets_value'] + 
            df['luxury_assets_value'] + 
            df['bank_asset_value']
        )
        
        df['loan_to_income_ratio'] = np.where(
            df['income_annum'] > 0,
            df['loan_amount'] / df['income_annum'],
            0
        )
        
        df['asset_to_loan_ratio'] = np.where(
            df['loan_amount'] > 0,
            df['total_assets'] / df['loan_amount'],
            0
        )
        
        # 標籤編碼
        self.le_edu = LabelEncoder()
        self.le_emp = LabelEncoder()
        
        self.le_edu.fit(EDUCATION_VALUES)
        self.le_emp.fit(EMPLOYMENT_VALUES)
        
        df['education_encoded'] = df['education'].apply(
            lambda x: self._safe_transform(self.le_edu, x, EDUCATION_VALUES)
        )
        df['self_employed_encoded'] = df['self_employed'].apply(
            lambda x: self._safe_transform(self.le_emp, x, EMPLOYMENT_VALUES)
        )
        
        print(f"✅ 特徵工程完成: {len(self.feature_cols)} 個特徵")
        
        return df
    
    def _safe_transform(self, encoder, value, valid_values):
        """安全的 LabelEncoder transform"""
        value = value.strip() if isinstance(value, str) else value
        if value not in valid_values:
            value = valid_values[0]
        return encoder.transform([value])[0]
    
    def calculate_interest_rate(self, df):
        """基於合理公式計算利率"""
        print("\n💰 計算利率標籤...")
        
        estimated_monthly_debt = df['loan_amount'] * 0.1 / 12
        monthly_income = df['income_annum'] / 12
        dti = np.where(monthly_income > 0, estimated_monthly_debt / monthly_income, 0.5)
        
        interest_rate = (
            INTEREST_RATE_BASE +
            (CREDIT_SCORE_MAX - df['cibil_score']) / CREDIT_SCORE_FACTOR +
            dti * DTI_RATE_MULTIPLIER +
            df['loan_to_income_ratio'] * LOAN_TO_INCOME_MULTIPLIER
        )
        
        interest_rate = np.clip(interest_rate, INTEREST_RATE_MIN, INTEREST_RATE_MAX)
        
        print(f"✅ 利率範圍: {interest_rate.min():.2f}% - {interest_rate.max():.2f}%")
        print(f"   平均利率: {interest_rate.mean():.2f}%")
        
        return interest_rate
    
    def feature_selection(self, X, y):
        """執行特徵選擇流程"""
        if not FEATURE_SELECTION_ENABLED:
            print("\n⏭️  特徵選擇已停用")
            self.selected_features = self.feature_cols
            return X, self.feature_cols
        
        # 執行綜合特徵選擇
        selected_feature_names = self.feature_selector.select_features(
            X, y, self.feature_cols, method='comprehensive'
        )
        
        # 更新特徵
        feature_indices = [self.feature_cols.index(f) for f in selected_feature_names]
        X_selected = X[:, feature_indices]
        
        # 驗證特徵獨立性
        validate_feature_independence(X_selected, selected_feature_names, threshold=0.8)
        
        self.selected_features = selected_feature_names
        
        return X_selected, selected_feature_names
    
    def create_models(self):
        """創建模型 (支援 XGBoost 和 Random Forest)"""
        
        if USE_XGBOOST and XGBOOST_AVAILABLE:
            print("\n🚀 使用 XGBoost (Gradient Boosting with L1/L2 Regularization)")
            
            clf = xgb.XGBClassifier(**XGBOOST_CLASSIFIER_PARAMS)
            reg = xgb.XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)
            
        else:
            print("\n🌲 使用 Random Forest")
            
            clf = RandomForestClassifier(**RANDOM_FOREST_CLASSIFIER_PARAMS)
            reg = RandomForestRegressor(**{
                'n_estimators': 100,
                'max_depth': 8,
                'min_samples_split': 5,
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            })
        
        return clf, reg
    
    def hyperparameter_tuning(self, X_train, y_train):
        """超參數調優"""
        if not HYPERPARAMETER_TUNING_ENABLED:
            print("\n⏭️  超參數調優已停用")
            return self.create_models()[0]
        
        print("\n" + "="*70)
        print("🎛️  超參數調優 (Optimization)")
        print("="*70)
        
        base_model = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ) if USE_XGBOOST and XGBOOST_AVAILABLE else RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        param_grid = XGBOOST_CLASSIFIER_PARAM_GRID if USE_XGBOOST and XGBOOST_AVAILABLE else {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        if TUNING_METHOD == 'random':
            print(f"使用 RandomizedSearchCV ({RANDOM_SEARCH_ITERATIONS} 次迭代)")
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=RANDOM_SEARCH_ITERATIONS,
                cv=CV_FOLDS,
                scoring=CV_SCORING,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )
        else:
            print(f"使用 GridSearchCV")
            search = GridSearchCV(
                base_model,
                param_grid=param_grid,
                cv=CV_FOLDS,
                scoring=CV_SCORING,
                n_jobs=-1,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        
        print(f"\n✅ 最佳參數:")
        for param, value in search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"\n✅ 最佳 {CV_SCORING} 分數: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def create_stacking_ensemble(self, X_train, y_train):
        """創建 Stacking 集成模型 (Boosting strategy)"""
        if not STACKING_ENABLED:
            return None
        
        print("\n" + "="*70)
        print("🎯 創建 Stacking 集成模型")
        print("="*70)
        
        # Base models
        estimators = [
            ('rf', RandomForestClassifier(**RANDOM_FOREST_CLASSIFIER_PARAMS)),
        ]
        
        if XGBOOST_AVAILABLE:
            estimators.append(
                ('xgb', xgb.XGBClassifier(**XGBOOST_CLASSIFIER_PARAMS))
            )
        
        # Meta learner (使用正則化的 Logistic Regression)
        meta_learner = LogisticRegression(
            penalty='l2',  # L2 regularization
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=STACKING_CV_FOLDS,
            n_jobs=-1
        )
        
        print("Base models:")
        for name, _ in estimators:
            print(f"   - {name}")
        print(f"Meta learner: Logistic Regression (L2 regularization, C=1.0)")
        
        return stacking_clf
    
    def train_models(self, df):
        """訓練模型的主流程"""
        print("\n" + "="*70)
        print("🚀 開始訓練模型")
        print("="*70)
        
        # 準備數據
        X = df[self.feature_cols].values
        y_clf = (df['loan_status'].str.strip() == 'Approved').astype(int)
        y_reg = self.calculate_interest_rate(df)
        
        print(f"\n📊 數據統計:")
        print(f"   總樣本: {len(X)}")
        print(f"   特徵數: {X.shape[1]}")
        print(f"   核准: {y_clf.sum()} ({y_clf.mean()*100:.1f}%)")
        print(f"   拒絕: {(1-y_clf).sum()} ({(1-y_clf.mean())*100:.1f}%)")
        
        # ===== 特徵選擇 (Representation Optimization) =====
        X, selected_features = self.feature_selection(X, y_clf)
        print(f"\n✅ 使用 {len(selected_features)} 個特徵進行訓練")
        
        # ===== 分割數據集 (Evaluation Strategy) =====
        # 先分出測試集 (20%)
        X_temp, X_test, y_clf_temp, y_clf_test, y_reg_temp, y_reg_test = \
            train_test_split(
                X, y_clf, y_reg,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_clf
            )
        
        # 再從臨時集分出訓練集和驗證集
        X_train, X_val, y_clf_train, y_clf_val, y_reg_train, y_reg_val = \
            train_test_split(
                X_temp, y_clf_temp, y_reg_temp,
                test_size=VALIDATION_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_clf_temp
            )
        
        print(f"\n📊 數據集分割 (Train/Val/Test):")
        print(f"   訓練集: {len(X_train)} 筆 ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   驗證集: {len(X_val)} 筆 ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   測試集: {len(X_test)} 筆 ({len(X_test)/len(X)*100:.1f}%)")
        
        # ===== 訓練分類器 =====
        print("\n" + "="*70)
        print("🎯 訓練分類模型")
        print("="*70)
        
        if HYPERPARAMETER_TUNING_ENABLED:
            self.clf = self.hyperparameter_tuning(X_train, y_clf_train)
        else:
            self.clf, _ = self.create_models()
            self.clf.fit(X_train, y_clf_train)
        
        # Stacking 集成 (可選)
        if USE_ENSEMBLE and STACKING_ENABLED:
            stacking_clf = self.create_stacking_ensemble(X_train, y_clf_train)
            print("\n訓練 Stacking 模型...")
            stacking_clf.fit(X_train, y_clf_train)
            
            # 比較單模型和集成模型
            single_score = self.clf.score(X_val, y_clf_val)
            stacking_score = stacking_clf.score(X_val, y_clf_val)
            
            print(f"\n驗證集準確率比較:")
            print(f"   單一模型: {single_score:.4f}")
            print(f"   Stacking: {stacking_score:.4f}")
            
            if stacking_score > single_score:
                print("✅ 使用 Stacking 模型")
                self.clf = stacking_clf
            else:
                print("✅ 使用單一模型")
        
        # ===== 訓練迴歸器 =====
        print("\n" + "="*70)
        print("💵 訓練利率預測模型")
        print("="*70)
        
        _, self.reg = self.create_models()
        self.reg.fit(X_train, y_reg_train)
        
        # ===== 評估模型 =====
        self._comprehensive_evaluation(
            X_train, X_val, X_test,
            y_clf_train, y_clf_val, y_clf_test,
            y_reg_train, y_reg_val, y_reg_test,
            selected_features
        )
        
        return X_test, y_clf_test, y_reg_test
    
    def _comprehensive_evaluation(self, X_train, X_val, X_test,
                                  y_clf_train, y_clf_val, y_clf_test,
                                  y_reg_train, y_reg_val, y_reg_test,
                                  feature_names):
        """全面評估 (訓練集、驗證集、測試集)"""
        
        print("\n" + "="*70)
        print("📊 全面模型評估 (Comprehensive Evaluation)")
        print("="*70)
        
        # ===== 分類器評估 =====
        print("\n【分類模型】")
        
        datasets = {
            'Train': (X_train, y_clf_train),
            'Validation': (X_val, y_clf_val),
            'Test': (X_test, y_clf_test)
        }
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\n{dataset_name} Set:")
            print("-" * 50)
            
            y_pred = self.clf.predict(X)
            y_prob = self.clf.predict_proba(X)[:, 1]
            
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_prob)
            
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"AUC-ROC:   {auc:.4f}")
            
            if dataset_name == 'Test':
                cm = confusion_matrix(y, y_pred)
                print(f"\nConfusion Matrix:")
                print(f"              預測拒絕  預測核准")
                print(f"實際拒絕        {cm[0,0]:4d}      {cm[0,1]:4d}")
                print(f"實際核准        {cm[1,0]:4d}      {cm[1,1]:4d}")
                
                self.metrics['classifier'] = {
                    'train_accuracy': accuracy_score(y_clf_train, self.clf.predict(X_train)),
                    'val_accuracy': accuracy_score(y_clf_val, self.clf.predict(X_val)),
                    'test_accuracy': float(acc),
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1_score': float(f1),
                    'auc_roc': float(auc),
                    'confusion_matrix': cm.tolist()
                }
        
        # 交叉驗證 (Cross-Validation)
        print("\n" + "-" * 70)
        print(f"{CV_FOLDS}-Fold Cross-Validation (Evaluation):")
        
        X_full_train = np.vstack([X_train, X_val])
        y_full_train = np.concatenate([y_clf_train, y_clf_val])
        
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_validate(
            self.clf, X_full_train, y_full_train,
            cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            return_train_score=True
        )
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            test_scores = cv_scores[f'test_{metric}']
            print(f"{metric:12s}: {test_scores.mean():.4f} (+/- {test_scores.std()*2:.4f})")
        
        self.metrics['classifier']['cv_accuracy'] = float(cv_scores['test_accuracy'].mean())
        self.metrics['classifier']['cv_std'] = float(cv_scores['test_accuracy'].std())
        
        # ===== 迴歸器評估 =====
        print("\n\n【利率預測模型】")
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\n{dataset_name} Set:")
            print("-" * 50)
            
            y_pred = self.reg.predict(X)
            
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            print(f"R²:   {r2:.4f}")
            print(f"MAE:  {mae:.4f}%")
            print(f"RMSE: {rmse:.4f}%")
            
            if dataset_name == 'Test':
                self.metrics['regressor'] = {
                    'train_r2': r2_score(y_reg_train, self.reg.predict(X_train)),
                    'val_r2': r2_score(y_reg_val, self.reg.predict(X_val)),
                    'test_r2': float(r2),
                    'mae': float(mae),
                    'rmse': float(rmse)
                }
        
        # ===== 找最佳閾值 (Threshold Optimization) =====
        self._find_optimal_threshold(X_test, y_clf_test)
        
        # ===== 特徵重要性 (Model Interpretability) =====
        self._analyze_feature_importance(feature_names)
        
        # ===== SHAP 值分析 (Model Explainability) =====
        if hasattr(self.clf, 'predict_proba'):
            try:
                self.explainer.explain_shap_values(
                    self.clf, X_test, feature_names, sample_size=200
                )
            except Exception as e:
                print(f"\n⚠️  SHAP 分析跳過: {str(e)}")
        
        # ===== Blessing of non-uniformity =====
        if hasattr(self.clf, 'feature_importances_'):
            importance = self.clf.feature_importances_
            self.dimensionality_reducer.apply_blessing_of_nonuniformity(
                X_test, importance
            )
    
    def _find_optimal_threshold(self, X_test, y_test):
        """使用 ROC 曲線找最佳閾值 (Threshold Optimization)"""
        print("\n" + "="*70)
        print("🎯 最佳決策閾值優化 (ROC Curve)")
        print("="*70)
        
        y_prob = self.clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        print(f"\n使用 Youden's J 統計量:")
        print(f"   最佳閾值: {self.optimal_threshold:.4f}")
        print(f"   TPR (召回率): {tpr[optimal_idx]:.4f}")
        print(f"   FPR (誤報率): {fpr[optimal_idx]:.4f}")
        print(f"   J = TPR - FPR = {j_scores[optimal_idx]:.4f}")
        
        y_pred_optimal = (y_prob >= self.optimal_threshold).astype(int)
        optimal_acc = accuracy_score(y_test, y_pred_optimal)
        optimal_prec = precision_score(y_test, y_pred_optimal)
        optimal_rec = recall_score(y_test, y_pred_optimal)
        
        print(f"\n使用最佳閾值的表現:")
        print(f"   Accuracy:  {optimal_acc:.4f}")
        print(f"   Precision: {optimal_prec:.4f}")
        print(f"   Recall:    {optimal_rec:.4f}")
        
        self.metrics['optimal_threshold'] = {
            'threshold': float(self.optimal_threshold),
            'accuracy': float(optimal_acc),
            'precision': float(optimal_prec),
            'recall': float(optimal_rec),
            'j_statistic': float(j_scores[optimal_idx])
        }
    
    def _analyze_feature_importance(self, feature_names):
        """分析特徵重要性 (Model Interpretability)"""
        self.explainer.explain_feature_importance(self.clf, feature_names, top_k=15)
        
        if self.explainer.feature_importance:
            self.metrics['feature_importance'] = self.explainer.feature_importance
    
    def save_models(self):
        """保存模型和相關資料"""
        print("\n" + "="*70)
        print("💾 保存模型")
        print("="*70)
        
        # 創建模型目錄
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        try:
            # 保存分類模型
            joblib.dump(self.clf, CLASSIFIER_PATH)
            print(f"✅ 分類模型: {CLASSIFIER_PATH}")
            
            # 保存迴歸模型
            joblib.dump(self.reg, REGRESSOR_PATH)
            print(f"✅ 迴歸模型: {REGRESSOR_PATH}")
            
            # 保存編碼器
            encoders = {
                'le_edu': self.le_edu,
                'le_emp': self.le_emp,
                'feature_cols': self.feature_cols,
                'selected_features': self.selected_features
            }
            joblib.dump(encoders, LABEL_ENCODERS_PATH)
            print(f"✅ 編碼器: {LABEL_ENCODERS_PATH}")
            
            # 準備指標字典
            self.metrics['optimal_threshold_value'] = float(self.optimal_threshold)
            self.metrics['selected_features'] = self.selected_features
            
            # 添加特徵選擇報告（如果可用）
            if hasattr(self.feature_selector, 'get_feature_selection_report'):
                try:
                    report = self.feature_selector.get_feature_selection_report()
                    if isinstance(report, dict):
                        # 確保所有值都是 JSON 可序列化的
                        serializable_report = {}
                        for key, value in report.items():
                            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                                serializable_report[key] = value
                            elif isinstance(value, np.ndarray):
                                serializable_report[key] = value.tolist()
                            else:
                                serializable_report[key] = str(value)
                        self.metrics['feature_selection_report'] = serializable_report
                except Exception as e:
                    print(f"⚠️  無法保存特徵選擇報告: {e}")
            
            # 確保所有 metrics 中的值都是可序列化的
            def make_serializable(obj):
                """遞迴確保物件可以被 JSON 序列化"""
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            serializable_metrics = make_serializable(self.metrics)
            
            # 保存為格式化的 JSON（便於閱讀和除錯）
            with open(MODEL_METRICS_PATH, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 評估指標: {MODEL_METRICS_PATH}")
            
            # 驗證 JSON 檔案
            with open(MODEL_METRICS_PATH, 'r', encoding='utf-8') as f:
                json.load(f)  # 嘗試讀取以驗證格式
            print("✅ JSON 格式驗證通過")
            
        except Exception as e:
            print(f"❌ 保存模型時發生錯誤: {str(e)}")
            raise
    
    def load_models(self):
        """載入已保存的模型"""
        print("📂 載入模型...")
        
        if not all([
            os.path.exists(CLASSIFIER_PATH),
            os.path.exists(REGRESSOR_PATH),
            os.path.exists(LABEL_ENCODERS_PATH)
        ]):
            print("⚠️  模型檔案不完整")
            return False
        
        try:
            # 載入模型
            self.clf = joblib.load(CLASSIFIER_PATH)
            self.reg = joblib.load(REGRESSOR_PATH)
            
            # 載入編碼器
            encoders = joblib.load(LABEL_ENCODERS_PATH)
            self.le_edu = encoders['le_edu']
            self.le_emp = encoders['le_emp']
            self.feature_cols = encoders['feature_cols']
            self.selected_features = encoders.get('selected_features', self.feature_cols)
            
            # 載入指標 (有錯誤處理)
            if os.path.exists(MODEL_METRICS_PATH):
                try:
                    with open(MODEL_METRICS_PATH, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # 確保檔案不是空的
                            self.metrics = json.loads(content)
                            self.optimal_threshold = self.metrics.get('optimal_threshold_value', APPROVAL_THRESHOLD)
                        else:
                            print("⚠️  model_metrics.json 是空的，使用預設值")
                            self.metrics = {}
                            self.optimal_threshold = APPROVAL_THRESHOLD
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️  model_metrics.json 格式錯誤: {e}")
                    print("   使用預設閾值")
                    self.metrics = {}
                    self.optimal_threshold = APPROVAL_THRESHOLD
            else:
                print("⚠️  model_metrics.json 不存在，使用預設值")
                self.metrics = {}
                self.optimal_threshold = APPROVAL_THRESHOLD
            
            print("✅ 模型載入成功")
            print(f"   使用閾值: {self.optimal_threshold:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {str(e)}")
            return False


def main():
    """主訓練流程"""
    print("\n" + "="*70)
    print("🎓 進階機器學習模型訓練系統")
    print("="*70)
    print("\n整合技術:")
    print("  ✓ 特徵選擇 (統計顯著性檢定、互信息)")
    print("  ✓ 交叉驗證 (K-Fold)")
    print("  ✓ 超參數調優 (Random/Grid Search)")
    print("  ✓ 正則化 (L1/L2 Regularization)")
    print("  ✓ 集成學習 (XGBoost, Stacking)")
    print("  ✓ 模型可解釋性 (SHAP)")
    print("  ✓ Blessing of non-uniformity")
    
    trainer = AdvancedLoanModelTrainer()
    
    df = trainer.load_and_prepare_data()
    df = trainer.feature_engineering(df)
    trainer.train_models(df)
    trainer.save_models()
    
    print("\n" + "="*70)
    print("🎉 訓練完成！")
    print("="*70)
    print(f"\n模型已保存至: {MODEL_DIR}/")
    print("\n可以開始使用 Streamlit 應用程式了！")


if __name__ == "__main__":
    main()
