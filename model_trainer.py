"""
貸款風險評估系統 - 模型訓練與評估模組
處理數據載入、特徵工程、模型訓練、評估和保存
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import *


class LoanModelTrainer:
    """貸款模型訓練器"""
    
    def __init__(self):
        self.clf = None
        self.reg = None
        self.le_edu = None
        self.le_emp = None
        self.feature_cols = FEATURE_COLUMNS
        self.metrics = {}
        self.optimal_threshold = APPROVAL_THRESHOLD
        
    def load_and_prepare_data(self, filepath=DATA_FILE):
        """載入並準備數據"""
        print(f"📂 載入數據: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到數據檔案: {filepath}")
        
        # 載入數據
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        # 檢查必要欄位
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"數據檔案缺少必要欄位: {', '.join(missing_cols)}")
        
        print(f"✅ 原始數據: {len(df)} 筆")
        
        # 清理數據
        df = df.dropna()
        print(f"✅ 清理後數據: {len(df)} 筆")
        
        if len(df) == 0:
            raise ValueError("清理後無有效數據")
        
        return df
    
    def feature_engineering(self, df):
        """特徵工程"""
        print("🔧 進行特徵工程...")
        
        df = df.copy()
        
        # 計算總資產
        df['total_assets'] = (
            df['residential_assets_value'] + 
            df['commercial_assets_value'] + 
            df['luxury_assets_value'] + 
            df['bank_asset_value']
        )
        
        # 計算比率（加入除零保護）
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
        
        # 標籤編碼（使用固定的類別值，避免未知類別）
        self.le_edu = LabelEncoder()
        self.le_emp = LabelEncoder()
        
        # 先 fit 所有可能的值
        self.le_edu.fit(EDUCATION_VALUES)
        self.le_emp.fit(EMPLOYMENT_VALUES)
        
        # 然後 transform（處理未知值）
        df['education_encoded'] = df['education'].apply(
            lambda x: self._safe_transform(self.le_edu, x, EDUCATION_VALUES)
        )
        df['self_employed_encoded'] = df['self_employed'].apply(
            lambda x: self._safe_transform(self.le_emp, x, EMPLOYMENT_VALUES)
        )
        
        print(f"✅ 特徵工程完成，共 {len(self.feature_cols)} 個特徵")
        
        return df
    
    def _safe_transform(self, encoder, value, valid_values):
        """安全的 LabelEncoder transform，處理未知值"""
        value = value.strip() if isinstance(value, str) else value
        # 如果值不在已知類別中，使用預設值
        if value not in valid_values:
            value = valid_values[0]  # 使用第一個有效值作為預設
        return encoder.transform([value])[0]
    
    def calculate_interest_rate(self, df):
        """
        基於合理的數學公式計算利率
        利率 = 基礎利率 + 信用評分影響 + 負債比影響 + 貸款金額影響
        """
        print("💰 計算合理利率...")
        
        # 計算負債收入比（估算）
        # 假設現有債務為貸款金額的10%
        estimated_monthly_debt = df['loan_amount'] * 0.1 / 12
        monthly_income = df['income_annum'] / 12
        dti = np.where(
            monthly_income > 0,
            estimated_monthly_debt / monthly_income,
            0.5  # 預設值
        )
        
        # 利率公式
        interest_rate = (
            INTEREST_RATE_BASE +
            # 信用評分影響：分數越高，利率越低
            (CREDIT_SCORE_MAX - df['cibil_score']) / CREDIT_SCORE_FACTOR +
            # 負債比影響：負債比越高，利率越高
            dti * DTI_RATE_MULTIPLIER +
            # 貸款金額影響：貸款金額/年收入越高，利率越高
            df['loan_to_income_ratio'] * LOAN_TO_INCOME_MULTIPLIER
        )
        
        # 限制在合理範圍內
        interest_rate = np.clip(interest_rate, INTEREST_RATE_MIN, INTEREST_RATE_MAX)
        
        print(f"✅ 利率範圍: {interest_rate.min():.2f}% - {interest_rate.max():.2f}%")
        print(f"   平均利率: {interest_rate.mean():.2f}%")
        
        return interest_rate
    
    def train_models(self, df):
        """訓練模型"""
        print("\n" + "="*60)
        print("🚀 開始訓練模型")
        print("="*60)
        
        # 準備特徵和標籤
        X = df[self.feature_cols]
        y_clf = (df['loan_status'].str.strip() == 'Approved').astype(int)
        y_reg = self.calculate_interest_rate(df)
        
        print(f"\n📊 數據統計:")
        print(f"   總樣本數: {len(X)}")
        print(f"   核准數量: {y_clf.sum()} ({y_clf.mean()*100:.1f}%)")
        print(f"   拒絕數量: {(1-y_clf).sum()} ({(1-y_clf.mean())*100:.1f}%)")
        
        # 分割訓練集和測試集
        X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = \
            train_test_split(
                X, y_clf, y_reg,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_clf  # 保持類別比例
            )
        
        print(f"\n📊 數據集分割:")
        print(f"   訓練集: {len(X_train)} 筆")
        print(f"   測試集: {len(X_test)} 筆")
        
        # 訓練分類模型
        print("\n🎯 訓練分類模型 (RandomForest)...")
        self.clf = RandomForestClassifier(**CLASSIFIER_PARAMS)
        self.clf.fit(X_train, y_clf_train)
        
        # 訓練迴歸模型
        print("💵 訓練利率預測模型 (RandomForest)...")
        self.reg = RandomForestRegressor(**REGRESSOR_PARAMS)
        self.reg.fit(X_train, y_reg_train)
        
        # 評估模型
        self._evaluate_classifier(X_train, X_test, y_clf_train, y_clf_test)
        self._evaluate_regressor(X_train, X_test, y_reg_train, y_reg_test)
        
        # 找最佳閾值
        self._find_optimal_threshold(X_test, y_clf_test)
        
        # 顯示特徵重要性
        self._show_feature_importance()
        
        print("\n✅ 模型訓練完成！")
        
        return X_test, y_clf_test, y_reg_test
    
    def _evaluate_classifier(self, X_train, X_test, y_train, y_test):
        """評估分類模型"""
        print("\n" + "="*60)
        print("📈 分類模型評估")
        print("="*60)
        
        # 訓練集表現
        y_train_pred = self.clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # 測試集表現
        y_test_pred = self.clf.predict(X_test)
        y_test_prob = self.clf.predict_proba(X_test)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_prob)
        
        print(f"\n訓練集準確率: {train_accuracy:.4f}")
        print(f"測試集準確率: {test_accuracy:.4f}")
        print(f"精確率 (Precision): {test_precision:.4f}")
        print(f"召回率 (Recall): {test_recall:.4f}")
        print(f"F1 分數: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        
        # 混淆矩陣
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\n混淆矩陣:")
        print(f"              預測拒絕  預測核准")
        print(f"實際拒絕        {cm[0,0]:4d}      {cm[0,1]:4d}")
        print(f"實際核准        {cm[1,0]:4d}      {cm[1,1]:4d}")
        
        # 儲存指標
        self.metrics['classifier'] = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(test_f1),
            'auc_roc': float(test_auc),
            'confusion_matrix': cm.tolist()
        }
        
        # 交叉驗證
        cv_scores = cross_val_score(
            self.clf, X_train, y_train, cv=5, scoring='accuracy'
        )
        print(f"\n5折交叉驗證準確率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        self.metrics['classifier']['cv_accuracy'] = float(cv_scores.mean())
    
    def _evaluate_regressor(self, X_train, X_test, y_train, y_test):
        """評估迴歸模型"""
        print("\n" + "="*60)
        print("📈 利率預測模型評估")
        print("="*60)
        
        # 訓練集表現
        y_train_pred = self.reg.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # 測試集表現
        y_test_pred = self.reg.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"\n訓練集 R² 分數: {train_r2:.4f}")
        print(f"測試集 R² 分數: {test_r2:.4f}")
        print(f"平均絕對誤差 (MAE): {test_mae:.4f}%")
        print(f"均方根誤差 (RMSE): {test_rmse:.4f}%")
        
        # 儲存指標
        self.metrics['regressor'] = {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'mae': float(test_mae),
            'rmse': float(test_rmse)
        }
    
    def _find_optimal_threshold(self, X_test, y_test):
        """使用 ROC 曲線找最佳閾值"""
        print("\n" + "="*60)
        print("🎯 尋找最佳決策閾值")
        print("="*60)
        
        y_prob = self.clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        # 使用 Youden's J statistic 找最佳閾值
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        print(f"\n使用 Youden's J 統計量:")
        print(f"   最佳閾值: {self.optimal_threshold:.4f}")
        print(f"   對應的 TPR (召回率): {tpr[optimal_idx]:.4f}")
        print(f"   對應的 FPR (誤報率): {fpr[optimal_idx]:.4f}")
        
        # 使用最佳閾值重新預測
        y_pred_optimal = (y_prob >= self.optimal_threshold).astype(int)
        optimal_accuracy = accuracy_score(y_test, y_pred_optimal)
        optimal_precision = precision_score(y_test, y_pred_optimal)
        optimal_recall = recall_score(y_test, y_pred_optimal)
        
        print(f"\n使用最佳閾值的表現:")
        print(f"   準確率: {optimal_accuracy:.4f}")
        print(f"   精確率: {optimal_precision:.4f}")
        print(f"   召回率: {optimal_recall:.4f}")
        
        self.metrics['optimal_threshold'] = {
            'threshold': float(self.optimal_threshold),
            'accuracy': float(optimal_accuracy),
            'precision': float(optimal_precision),
            'recall': float(optimal_recall)
        }
    
    def _show_feature_importance(self):
        """顯示特徵重要性"""
        print("\n" + "="*60)
        print("📊 特徵重要性 (Top 10)")
        print("="*60)
        
        importance = self.clf.feature_importances_
        indices = np.argsort(importance)[::-1][:10]
        
        print(f"\n{'排名':<5} {'特徵名稱':<30} {'重要性':<10}")
        print("-" * 50)
        for i, idx in enumerate(indices, 1):
            print(f"{i:<5} {self.feature_cols[idx]:<30} {importance[idx]:.4f}")
        
        self.metrics['feature_importance'] = {
            self.feature_cols[i]: float(importance[i]) 
            for i in indices
        }
    
    def save_models(self):
        """保存模型和編碼器"""
        print("\n" + "="*60)
        print("💾 保存模型")
        print("="*60)
        
        # 創建模型目錄
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 保存模型
        joblib.dump(self.clf, CLASSIFIER_PATH)
        print(f"✅ 分類模型已保存: {CLASSIFIER_PATH}")
        
        joblib.dump(self.reg, REGRESSOR_PATH)
        print(f"✅ 迴歸模型已保存: {REGRESSOR_PATH}")
        
        # 保存 LabelEncoders
        encoders = {
            'le_edu': self.le_edu,
            'le_emp': self.le_emp,
            'feature_cols': self.feature_cols
        }
        joblib.dump(encoders, LABEL_ENCODERS_PATH)
        print(f"✅ 編碼器已保存: {LABEL_ENCODERS_PATH}")
        
        # 保存模型評估指標和最佳閾值
        self.metrics['optimal_threshold_value'] = float(self.optimal_threshold)
        with open(MODEL_METRICS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ 模型指標已保存: {MODEL_METRICS_PATH}")
    
    def load_models(self):
        """載入已保存的模型"""
        print("📂 載入已保存的模型...")
        
        if not all([
            os.path.exists(CLASSIFIER_PATH),
            os.path.exists(REGRESSOR_PATH),
            os.path.exists(LABEL_ENCODERS_PATH),
            os.path.exists(MODEL_METRICS_PATH)
        ]):
            return False
        
        self.clf = joblib.load(CLASSIFIER_PATH)
        self.reg = joblib.load(REGRESSOR_PATH)
        
        encoders = joblib.load(LABEL_ENCODERS_PATH)
        self.le_edu = encoders['le_edu']
        self.le_emp = encoders['le_emp']
        self.feature_cols = encoders['feature_cols']
        
        with open(MODEL_METRICS_PATH, 'r', encoding='utf-8') as f:
            self.metrics = json.load(f)
            self.optimal_threshold = self.metrics.get('optimal_threshold_value', APPROVAL_THRESHOLD)
        
        print("✅ 模型載入成功！")
        print(f"   最佳閾值: {self.optimal_threshold:.4f}")
        print(f"   測試集準確率: {self.metrics['classifier']['test_accuracy']:.4f}")
        print(f"   AUC-ROC: {self.metrics['classifier']['auc_roc']:.4f}")
        
        return True


def main():
    """主訓練流程"""
    trainer = LoanModelTrainer()
    
    # 載入數據
    df = trainer.load_and_prepare_data()
    
    # 特徵工程
    df = trainer.feature_engineering(df)
    
    # 訓練模型
    trainer.train_models(df)
    
    # 保存模型
    trainer.save_models()
    
    print("\n" + "="*60)
    print("🎉 訓練完成！")
    print("="*60)
    print(f"\n模型和相關文件已保存至: {MODEL_DIR}/")
    print("\n可以開始使用 Streamlit 應用程式了！")


if __name__ == "__main__":
    main()
