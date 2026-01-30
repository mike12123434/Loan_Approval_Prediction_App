"""
個人信貸申請試算系統 - 重構優化版
使用預訓練模型、最佳閾值和基於數據驗證的規則
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings

from config import *
from model_trainer import LoanModelTrainer
from assessment import (
    calculate_credit_score, 
    get_hybrid_assessment,
    calculate_monthly_payment,
    get_improvement_suggestions,
    estimate_assets
)

warnings.filterwarnings('ignore')

# ==================== 頁面設置 ====================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="centered"
)

# CSS 樣式
st.markdown("""
<style>
    .main {
        max-width: 800px; 
        margin: 0 auto; 
        font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif;
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2c5282;
        color: white;
        border: none;
        padding: 12px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 600;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #2a4365;
    }
    .result-box {
        padding: 24px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .approved {
        background-color: #e6fffa;
        border-left: 5px solid #38b2ac;
    }
    .rejected {
        background-color: #fff5f5;
        border-left: 5px solid #fc8181;
    }
    .info-box {
        background-color: #ebf8ff;
        border-left: 5px solid #4299e1;
        padding: 15px;
        border-radius: 6px;
        margin: 15px 0;
    }
    h1 {
        color: #2d3748;
        font-size: 2rem;
        text-align: center;
        margin-bottom: 10px;
    }
    h2 {
        color: #4a5568;
        font-size: 1.3rem;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
    }
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 0.95rem;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f7fafc;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        margin-top: 5px;
    }
    .suggestion-card {
        background-color: #f7fafc;
        border-left: 4px solid #4299e1;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .suggestion-title {
        font-weight: 600;
        color: #2d3748;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }
    hr {
        margin: 30px 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 模型初始化 ====================

@st.cache_resource
def init_models():
    """初始化模型（載入或訓練）"""
    trainer = LoanModelTrainer()
    
    # 嘗試載入已保存的模型
    if trainer.load_models():
        return (
            trainer.clf, 
            trainer.reg, 
            trainer.le_edu, 
            trainer.le_emp, 
            trainer.feature_cols,
            trainer.optimal_threshold,
            trainer.metrics
        )
    
    # 如果沒有保存的模型，顯示錯誤訊息
    st.error("""
    ❌ 找不到預訓練模型！
    
    請先執行以下指令訓練模型：
    ```bash
    python model_trainer.py
    ```
    
    或確認以下檔案存在：
    - models/loan_classifier.pkl
    - models/rate_regressor.pkl
    - models/label_encoders.pkl
    - models/model_metrics.json
    """)
    return None

# ==================== 主程式 ====================

def main():
    """主程式"""
    
    # 標題
    st.markdown("# 💰 個人信貸試算系統")
    st.markdown('<p class="subtitle">AI 智能評估 · 快速試算 · 專業建議</p>', 
                unsafe_allow_html=True)
    
    # 初始化模型
    model_data = init_models()
    if model_data is None:
        st.stop()
    
    clf, reg, le_edu, le_emp, feature_cols, optimal_threshold, metrics = model_data
    
    # 顯示模型資訊
    with st.expander("ℹ️ 模型資訊", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("模型準確率", f"{metrics['classifier']['test_accuracy']:.2%}")
        with col2:
            st.metric("AUC-ROC", f"{metrics['classifier']['auc_roc']:.3f}")
        with col3:
            st.metric("最佳閾值", f"{optimal_threshold:.3f}")
        
        st.markdown(f"""
        **模型說明：**
        - 使用 Random Forest 演算法
        - 訓練樣本數：根據實際數據訓練
        - 特徵數量：{len(feature_cols)} 個
        - 最佳閾值通過 ROC 曲線優化得出
        """)
    
    # ==================== 輸入區域 ====================
    
    st.markdown("## 📝 申請資料")
    
    # 收入與貸款資訊
    col1, col2 = st.columns(2)
    
    with col1:
        mon_income = st.number_input(
            "月收入（稅後）",
            min_value=INCOME_MIN,
            max_value=INCOME_MAX,
            value=INCOME_DEFAULT,
            step=5000,
            format="%d",
            help="您的稅後月收入"
        )
    
    with col2:
        loan_amt = st.number_input(
            "申請貸款金額",
            min_value=LOAN_MIN,
            max_value=LOAN_MAX,
            value=LOAN_DEFAULT,
            step=50000,
            format="%d",
            help="您希望申請的貸款總額"
        )
    
    # 支出與債務
    col3, col4 = st.columns(2)
    
    with col3:
        mon_expense = st.number_input(
            "每月固定支出",
            min_value=0,
            max_value=mon_income,
            value=min(30000, int(mon_income * 0.5)),
            step=1000,
            format="%d",
            help="包含房租、水電、生活費等"
        )
    
    with col4:
        cur_debt = st.number_input(
            "現有貸款月付金",
            min_value=0,
            max_value=mon_income,
            value=0,
            step=1000,
            format="%d",
            help="目前其他貸款的每月還款金額"
        )
    
    # 信用狀況
    col5, col6 = st.columns(2)
    
    with col5:
        cc_limit = st.number_input(
            "信用卡總額度",
            min_value=0,
            max_value=1000000,
            value=100000,
            step=10000,
            format="%d",
            help="所有信用卡的總額度"
        )
    
    with col6:
        term_yr = st.selectbox(
            "還款期限（年）",
            options=TERM_OPTIONS,
            index=TERM_OPTIONS.index(TERM_DEFAULT),
            help="貸款期限"
        )
    
    # 個人資料
    with st.expander("📋 基本資料"):
        dependents = st.number_input(
            "扶養人數",
            min_value=0,
            max_value=10,
            value=0,
            help="需要您經濟支持的家人數量"
        )
        
        edu = st.selectbox(
            "最高學歷",
            options=EDUCATION_VALUES,
            format_func=lambda x: "大學（含）以上" if "Grad" in x else "高中職（含）以下"
        )
        
        self_emp = st.selectbox(
            "工作類型",
            options=EMPLOYMENT_VALUES,
            format_func=lambda x: "一般受薪" if x == " No" else "自僱／接案"
        )
    
    # 進階設定
    with st.expander("⚙️ 進階設定"):
        use_rules = st.checkbox(
            "使用混合評估（AI + 規則）",
            value=True,
            help="結合 AI 模型與基於數據驗證的業務規則"
        )
        
        show_details = st.checkbox(
            "顯示詳細評估過程",
            value=False,
            help="顯示特徵值、AI預測等詳細資訊"
        )
    
    # ==================== 提交評估 ====================
    
    if st.button("🚀 提交評估"):
        
        # 輸入驗證
        if mon_income <= 0:
            st.error("❌ 請輸入有效的月收入")
            st.stop()
        
        if loan_amt <= 0:
            st.error("❌ 請輸入有效的貸款金額")
            st.stop()
        
        if mon_expense > mon_income:
            st.warning("⚠️ 您的支出超過收入，這將嚴重影響貸款評估")
        
        with st.spinner("🔍 AI 正在評估中..."):
            
            # ==================== 計算財務指標 ====================
            
            # 計算負債收入比（DTI）
            estimated_cc_payment = cc_limit * CREDIT_CARD_USAGE_RATE
            total_monthly_debt = cur_debt + estimated_cc_payment
            dti = total_monthly_debt / mon_income if mon_income > 0 else 1.0
            
            # 計算信用評分
            credit_score = calculate_credit_score(mon_income, dti, dependents, self_emp)
            
            # 準備特徵向量
            ann_income = mon_income * 12
            
            # 估算資產
            est_residential, est_commercial, est_luxury, est_bank, total_assets = \
                estimate_assets(ann_income, self_emp)
            
            # 計算比率
            loan_to_income = loan_amt / ann_income if ann_income > 0 else 0
            asset_to_loan = total_assets / loan_amt if loan_amt > 0 else 0
            
            # ==================== 建立特徵陣列 ====================
            
            try:
                # 安全的 LabelEncoder transform
                edu_encoded = le_edu.transform([edu])[0] if edu in EDUCATION_VALUES else 0
                emp_encoded = le_emp.transform([self_emp])[0] if self_emp in EMPLOYMENT_VALUES else 0
                
                features = np.array([[
                    dependents,
                    edu_encoded,
                    emp_encoded,
                    ann_income,
                    loan_amt,
                    term_yr * 12,
                    credit_score,
                    est_residential,
                    est_commercial,
                    est_luxury,
                    est_bank,
                    total_assets,
                    loan_to_income,
                    asset_to_loan
                ]])
            except Exception as e:
                st.error(f"❌ 特徵處理錯誤: {str(e)}")
                st.stop()
            
            # ==================== AI 模型預測 ====================
            
            try:
                ai_prob = clf.predict_proba(features)[0][1]
            except Exception as e:
                st.error(f"❌ 模型預測錯誤: {str(e)}")
                st.stop()
            
            # 混合評估
            final_prob, reason, rule_triggered = get_hybrid_assessment(
                ai_prob, credit_score, dti, loan_to_income, 
                optimal_threshold, use_rules
            )
            
            # 預測利率
            try:
                pred_rate = reg.predict(features)[0]
                pred_rate = max(INTEREST_RATE_MIN, min(INTEREST_RATE_MAX, pred_rate))
            except Exception as e:
                pred_rate = 6.5  # 預設利率
            
            # 計算月付金
            monthly_payment = calculate_monthly_payment(loan_amt, pred_rate, term_yr * 12)
            
            # ==================== 顯示詳細資訊 ====================
            
            if show_details:
                st.markdown("---")
                st.markdown("### 🔍 評估詳情")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**輸入特徵：**")
                    st.json({
                        "年收入": f"NT$ {ann_income:,}",
                        "貸款金額": f"NT$ {loan_amt:,}",
                        "信用評分": int(credit_score),
                        "負債收入比": f"{dti:.2%}",
                        "貸款/收入比": f"{loan_to_income:.2f}",
                    })
                
                with detail_col2:
                    st.markdown("**AI 預測：**")
                    st.json({
                        "AI 原始機率": f"{ai_prob:.2%}",
                        "最佳閾值": f"{optimal_threshold:.2%}",
                        "規則觸發": "是" if rule_triggered else "否",
                        "最終機率": f"{final_prob:.2%}",
                    })
            
            # ==================== 顯示結果 ====================
            
            st.markdown("---")
            st.markdown("## 📊 評估結果")
            
            # 評估指標
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{int(credit_score)}</div>', 
                           unsafe_allow_html=True)
                st.markdown('<div class="metric-label">模擬信用評分</div>', 
                           unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{dti:.1%}</div>', 
                           unsafe_allow_html=True)
                st.markdown('<div class="metric-label">負債收入比</div>', 
                           unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{final_prob:.1%}</div>', 
                           unsafe_allow_html=True)
                st.markdown('<div class="metric-label">核准機率</div>', 
                           unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 核准 / 拒絕結果
            is_approved = final_prob >= optimal_threshold
            
            if is_approved:
                # 核准
                st.markdown(f'''
                <div class="result-box approved">
                    <h3 style="margin:0; color:#2d3748;">✅ 初步評估：建議核准</h3>
                    <p style="margin:10px 0 0 0; color:#4a5568;">
                        <strong>判定依據：</strong>{reason}
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                # 貸款條件
                st.markdown("### 💰 預估貸款條件")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric("預估年利率", f"{pred_rate:.2f}%")
                    st.metric("貸款金額", f"NT$ {loan_amt:,}")
                
                with result_col2:
                    st.metric("預估月付金", f"NT$ {int(monthly_payment):,}")
                    st.metric("還款期限", f"{term_yr} 年")
                
                # 還款摘要
                total_payment = monthly_payment * term_yr * 12
                total_interest = total_payment - loan_amt
                
                st.markdown("### 📈 還款摘要")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("還款總額", f"NT$ {int(total_payment):,}")
                with summary_col2:
                    st.metric("總利息", f"NT$ {int(total_interest):,}")
                with summary_col3:
                    st.metric("利息佔比", f"{(total_interest/loan_amt*100):.1f}%")
                
            else:
                # 拒絕
                st.markdown(f'''
                <div class="result-box rejected">
                    <h3 style="margin:0; color:#2d3748;">❌ 初步評估：建議婉拒</h3>
                    <p style="margin:10px 0 0 0; color:#4a5568;">
                        <strong>判定依據：</strong>{reason}
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                # 改善建議
                st.markdown("### 💡 改善建議")
                
                suggestions = get_improvement_suggestions(
                    dti, credit_score, mon_expense, mon_income,
                    loan_amt, ann_income, dependents
                )
                
                for suggestion in suggestions:
                    st.markdown(f"""
                    <div class="suggestion-card">
                        <div class="suggestion-title">{suggestion['icon']} {suggestion['title']}</div>
                        <p style="margin: 8px 0; color: #4a5568;">{suggestion['content']}</p>
                        <ul style="margin: 8px 0; padding-left: 20px; color: #4a5568;">
                    """, unsafe_allow_html=True)
                    
                    for action in suggestion['actions']:
                        st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # 財務健康度
            st.markdown("---")
            st.markdown("### 📋 財務健康分析")
            
            net_disposable = mon_income - mon_expense - cur_debt - estimated_cc_payment
            
            health_col1, health_col2 = st.columns(2)
            
            with health_col1:
                st.write("**月收入**")
                st.write(f"NT$ {mon_income:,}")
                
                st.write("**月支出**")
                st.write(f"NT$ {mon_expense:,}")
                
                st.write("**現有貸款**")
                st.write(f"NT$ {cur_debt:,}")
            
            with health_col2:
                st.write("**預估信用卡費用**")
                st.write(f"NT$ {int(estimated_cc_payment):,}")
                
                st.write("**淨可支配所得**")
                disposable_color = "green" if net_disposable > 0 else "red"
                st.markdown(f"<span style='color:{disposable_color}; font-weight:600;'>NT$ {int(net_disposable):,}</span>", 
                           unsafe_allow_html=True)
                
                if is_approved:
                    st.write("**扣除新貸款後**")
                    remaining = net_disposable - monthly_payment
                    remaining_color = "green" if remaining > 0 else "red"
                    st.markdown(f"<span style='color:{remaining_color}; font-weight:600;'>NT$ {int(remaining):,}</span>", 
                               unsafe_allow_html=True)
                    
                    if remaining < 0:
                        st.warning("⚠️ 扣除新貸款後，您的可支配所得為負，建議重新評估還款能力")
    
    # 頁尾
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #a0aec0; font-size: 0.85rem; padding: 20px;'>
        <p><strong>⚠️ 重要聲明</strong></p>
        <p>本系統僅供初步試算參考，實際核貸結果以金融機構審核為準。</p>
        <p>建議與專業理財顧問諮詢，以獲得更詳細的評估與規劃。</p>
        <p style='margin-top: 15px; font-size: 0.8rem;'>
            AI 模型準確率: {:.2%} | AUC-ROC: {:.3f} | 最佳閾值: {:.3f}
        </p>
    </div>
    """.format(
        metrics['classifier']['test_accuracy'],
        metrics['classifier']['auc_roc'],
        optimal_threshold
    ), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
