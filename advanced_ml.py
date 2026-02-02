"""
進階機器學習模組
- 特徵選擇與統計顯著性檢定
- 模型可解釋性 (SHAP)
- 維度災難應對策略
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    特徵選擇器
    - 移除高度相關特徵 (多重共線性)
    - 統計顯著性檢定
    - 互信息選擇
    """
    
    def __init__(self, correlation_threshold=0.95, p_value_threshold=0.05):
        self.correlation_threshold = correlation_threshold
        self.p_value_threshold = p_value_threshold
        self.selected_features = None
        self.correlation_matrix = None
        self.p_values = None
        self.mutual_info_scores = None
        
    def remove_correlated_features(self, X, feature_names):
        """
        移除高度相關的特徵 (應對多重共線性)
        使用 Blessing of non-uniformity: 保留資訊量更大的特徵
        """
        print(f"\n🔍 檢測高度相關特徵 (閾值 > {self.correlation_threshold})...")
        
        # 計算相關係數矩陣
        corr_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()
        self.correlation_matrix = corr_matrix
        
        # 找出高度相關的特徵對
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = set()
        for column in upper_triangle.columns:
            correlated_features = upper_triangle.index[
                upper_triangle[column] > self.correlation_threshold
            ].tolist()
            
            if correlated_features:
                # 保留方差較大的特徵 (資訊量較多)
                variances = {
                    column: X[:, feature_names.index(column)].var(),
                    **{feat: X[:, feature_names.index(feat)].var() 
                       for feat in correlated_features}
                }
                features_sorted = sorted(variances.items(), key=lambda x: x[1], reverse=True)
                
                # 移除方差較小的
                for feat, _ in features_sorted[1:]:
                    if feat not in to_drop:
                        to_drop.add(feat)
                        print(f"   移除 {feat} (與 {column} 相關性 = {corr_matrix.loc[feat, column]:.3f})")
        
        selected_features = [f for f in feature_names if f not in to_drop]
        print(f"✅ 移除了 {len(to_drop)} 個高度相關特徵，保留 {len(selected_features)} 個")
        
        return selected_features
    
    def statistical_significance_test(self, X, y, feature_names):
        """
        統計顯著性檢定 (ANOVA F-test)
        檢驗特徵與目標變數是否有統計上的顯著關聯
        """
        print(f"\n📊 執行統計顯著性檢定 (p-value < {self.p_value_threshold})...")
        
        # ANOVA F-test
        f_scores, p_values = f_classif(X, y)
        self.p_values = dict(zip(feature_names, p_values))
        
        # 選擇統計顯著的特徵
        significant_features = []
        for i, (feature, p_value) in enumerate(zip(feature_names, p_values)):
            if p_value < self.p_value_threshold:
                significant_features.append(feature)
                print(f"   ✓ {feature}: p-value = {p_value:.4e} (F-score = {f_scores[i]:.2f})")
            else:
                print(f"   ✗ {feature}: p-value = {p_value:.4e} (不顯著)")
        
        print(f"✅ {len(significant_features)}/{len(feature_names)} 個特徵通過顯著性檢定")
        
        return significant_features
    
    def mutual_information_selection(self, X, y, feature_names, top_k=None):
        """
        互信息選擇
        衡量特徵與目標之間的依賴性 (包含非線性關係)
        """
        print(f"\n🔗 計算互信息分數...")
        
        # 計算互信息
        mi_scores = mutual_info_classif(X, y, random_state=42)
        self.mutual_info_scores = dict(zip(feature_names, mi_scores))
        
        # 排序並顯示
        sorted_features = sorted(
            zip(feature_names, mi_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\n互信息分數排名:")
        for i, (feature, score) in enumerate(sorted_features[:15], 1):
            print(f"   {i:2d}. {feature:<30} {score:.4f}")
        
        if top_k:
            selected_features = [f for f, _ in sorted_features[:top_k]]
            print(f"\n✅ 選擇互信息分數前 {top_k} 的特徵")
            return selected_features
        
        return feature_names
    
    def select_features(self, X, y, feature_names, method='comprehensive'):
        """
        綜合特徵選擇流程
        
        method:
            - 'correlation': 只移除相關特徵
            - 'statistical': 只做統計檢定
            - 'mutual_info': 只用互信息
            - 'comprehensive': 綜合所有方法
        """
        print("="*70)
        print("🎯 特徵選擇流程")
        print("="*70)
        
        current_features = feature_names.copy()
        
        if method in ['correlation', 'comprehensive']:
            # 步驟 1: 移除高度相關特徵
            current_features = self.remove_correlated_features(X, current_features)
            feature_indices = [feature_names.index(f) for f in current_features]
            X = X[:, feature_indices]
        
        if method in ['statistical', 'comprehensive']:
            # 步驟 2: 統計顯著性檢定
            current_features = self.statistical_significance_test(
                X, y, current_features
            )
            feature_indices = [
                [f for f in feature_names if f in current_features].index(feat)
                for feat in current_features
            ]
            X = X[:, feature_indices]
        
        if method in ['mutual_info', 'comprehensive']:
            # 步驟 3: 互信息分析 (不做選擇，只做分析)
            self.mutual_information_selection(X, y, current_features)
        
        self.selected_features = current_features
        
        print("\n" + "="*70)
        print(f"✅ 特徵選擇完成: {len(feature_names)} → {len(current_features)} 個特徵")
        print("="*70)
        
        return current_features
    
    def get_feature_selection_report(self):
        """生成特徵選擇報告"""
        if self.selected_features is None:
            return "尚未執行特徵選擇"
        
        report = {
            'selected_features': self.selected_features,
            'n_selected': len(self.selected_features),
            'p_values': self.p_values,
            'mutual_info_scores': self.mutual_info_scores
        }
        
        return report


class DimensionalityReducer:
    """
    維度災難應對策略
    - Blessing of non-uniformity: 利用數據的非均勻分佈
    - 特徵重要性加權
    """
    
    def __init__(self):
        self.feature_weights = None
        
    def apply_blessing_of_nonuniformity(self, X, feature_importance):
        """
        應用 Blessing of non-uniformity
        根據特徵重要性對特徵進行加權，讓模型專注於重要特徵
        """
        print("\n🌟 應用 Blessing of non-uniformity...")
        
        # 正規化重要性分數
        weights = feature_importance / feature_importance.sum()
        self.feature_weights = weights
        
        # 計算有效維度 (effective dimensionality)
        # 使用 Shannon entropy
        entropy = -np.sum(weights * np.log2(weights + 1e-10))
        effective_dim = 2 ** entropy
        
        print(f"   原始維度: {len(feature_importance)}")
        print(f"   有效維度: {effective_dim:.2f}")
        print(f"   維度降低: {(1 - effective_dim/len(feature_importance))*100:.1f}%")
        
        # 顯示前 10 個最重要特徵的權重分佈
        top_10_weight = weights[:10].sum()
        print(f"   前 10 特徵佔總權重: {top_10_weight*100:.1f}%")
        
        return weights


class ModelExplainer:
    """
    模型可解釋性工具
    - 特徵重要性分析
    - 部分依賴圖 (Partial Dependence)
    - SHAP 值 (如果安裝)
    """
    
    def __init__(self):
        self.feature_importance = None
        self.shap_values = None
        
    def explain_feature_importance(self, model, feature_names, top_k=15):
        """解釋特徵重要性"""
        print("\n" + "="*70)
        print("📊 特徵重要性分析")
        print("="*70)
        
        # 獲取特徵重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            print("模型不支援 feature_importances_")
            return None
        
        self.feature_importance = dict(zip(feature_names, importance))
        
        # 排序
        sorted_importance = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\n{'排名':<5} {'特徵名稱':<30} {'重要性':<10} {'累積佔比':<10}")
        print("-" * 70)
        
        cumsum = 0
        for i, (feature, score) in enumerate(sorted_importance[:top_k], 1):
            cumsum += score
            print(f"{i:<5} {feature:<30} {score:.4f}     {cumsum:.2%}")
        
        print("\n" + "="*70)
        
        return self.feature_importance
    
    def explain_shap_values(self, model, X, feature_names, sample_size=100):
        """
        使用 SHAP 解釋模型 (需要安裝 shap 套件)
        SHAP (SHapley Additive exPlanations) 提供更精確的特徵貢獻度
        """
        try:
            import shap
            
            print("\n" + "="*70)
            print("🔍 SHAP 值分析")
            print("="*70)
            
            # 選擇樣本
            sample_indices = np.random.choice(
                len(X), 
                min(sample_size, len(X)), 
                replace=False
            )
            X_sample = X[sample_indices]
            
            # 創建 explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # 如果是分類問題，取正類的 SHAP 值
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            self.shap_values = shap_values
            
            # 計算平均絕對 SHAP 值
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            sorted_shap = sorted(
                zip(feature_names, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"\n{'排名':<5} {'特徵名稱':<30} {'平均|SHAP|':<12}")
            print("-" * 70)
            
            for i, (feature, score) in enumerate(sorted_shap[:15], 1):
                print(f"{i:<5} {feature:<30} {score:.6f}")
            
            print("\n✅ SHAP 分析完成")
            print("="*70)
            
            return shap_values
            
        except ImportError:
            print("\n⚠️  未安裝 shap 套件，跳過 SHAP 分析")
            print("   安裝方式: pip install shap")
            return None
    
    def generate_explanation_report(self):
        """生成可解釋性報告"""
        report = {
            'feature_importance': self.feature_importance,
            'shap_available': self.shap_values is not None
        }
        
        return report


def validate_feature_independence(X, feature_names, threshold=0.8):
    """
    驗證特徵獨立性
    檢查特徵之間是否過度相關
    """
    print("\n" + "="*70)
    print("🔍 特徵獨立性驗證")
    print("="*70)
    
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
    
    # 找出高度相關的特徵對
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > threshold:
                high_corr_pairs.append((
                    feature_names[i], 
                    feature_names[j], 
                    corr
                ))
    
    if high_corr_pairs:
        print(f"\n⚠️  發現 {len(high_corr_pairs)} 組高度相關特徵 (|r| > {threshold}):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"   {feat1} ↔ {feat2}: {corr:.3f}")
        print("\n建議: 考慮移除其中一個特徵或進行特徵組合")
    else:
        print(f"\n✅ 所有特徵相關性 < {threshold}，特徵獨立性良好")
    
    print("="*70)
    
    return high_corr_pairs
