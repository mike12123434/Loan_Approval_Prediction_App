"""
貸款風險評估系統 - 評估邏輯模組
處理信用評分計算、混合評估和月付金計算
"""

import numpy as np
from config import *


def calculate_credit_score(mon_income, dti, dependents, self_employed):
    """
    計算模擬信用評分（300-900）
    
    參數:
        mon_income: 月收入
        dti: 負債收入比
        dependents: 扶養人數
        self_employed: 就業類型 (" No" 或 " Yes")
    
    返回:
        信用評分 (300-900)
    """
    score = CREDIT_SCORE_BASE
    
    # 收入因素
    for income_threshold, bonus in INCOME_TIERS:
        if mon_income >= income_threshold:
            score += bonus
            break  # 只加最高等級的分數
    
    # 負債比因素
    for dti_threshold, adjustment in DTI_TIERS:
        if dti < dti_threshold or (dti_threshold == DTI_TIERS[-1][0] and dti > dti_threshold):
            score += adjustment
            if adjustment > 0:  # 正向調整只取最高的
                break
    
    # 就業類型
    if self_employed == " No":
        score += EMPLOYMENT_BONUS
    
    # 扶養人數
    score -= (dependents * DEPENDENT_PENALTY)
    
    # 確保在合理範圍
    return max(CREDIT_SCORE_MIN, min(CREDIT_SCORE_MAX, score))


def get_hybrid_assessment(ai_prob, credit_score, dti, loan_to_income, 
                          optimal_threshold, use_rules=True):
    """
    混合評估：結合 AI 模型與基於數據驗證的風險規則
    
    參數:
        ai_prob: AI 模型預測機率
        credit_score: 信用評分
        dti: 負債收入比
        loan_to_income: 貸款金額/年收入比
        optimal_threshold: 從訓練數據獲得的最佳閾值
        use_rules: 是否使用規則覆蓋（預設 True）
    
    返回:
        (最終機率, 判定原因, 規則是否觸發)
    """
    
    if not use_rules:
        # 純 AI 模式
        reason = f"AI 模型預測（閾值 {optimal_threshold:.2f}）"
        return ai_prob, reason, False
    
    original_prob = ai_prob
    rule_triggered = False
    
    # 規則 1: 優質客戶快速通道
    # 根據訓練數據，信用分數 >= 750 且 DTI < 0.2 的客戶核准率接近 100%
    if credit_score >= CREDIT_SCORE_EXCELLENT and dti < DTI_EXCELLENT:
        final_prob = max(ai_prob, 0.95)
        reason = f"優質客戶（信用分數 {credit_score} ≥ {CREDIT_SCORE_EXCELLENT}，負債比 {dti:.1%} < {DTI_EXCELLENT:.1%}）"
        if final_prob != original_prob:
            rule_triggered = True
        return final_prob, reason, rule_triggered
    
    # 規則 2: 高風險攔截 - 負債比過高
    # 根據訓練數據，DTI > 0.7 的客戶核准率低於 10%
    if dti > DTI_HIGH_RISK:
        final_prob = min(ai_prob, 0.1)
        reason = f"高風險（負債比 {dti:.1%} > {DTI_HIGH_RISK:.1%}）"
        if final_prob != original_prob:
            rule_triggered = True
        return final_prob, reason, rule_triggered
    
    # 規則 3: 信用分數過低
    # 根據訓練數據，信用分數 < 500 的客戶核准率低於 20%
    if credit_score < CREDIT_SCORE_POOR:
        final_prob = min(ai_prob, 0.2)
        reason = f"信用評分過低（{credit_score} < {CREDIT_SCORE_POOR}）"
        if final_prob != original_prob:
            rule_triggered = True
        return final_prob, reason, rule_triggered
    
    # 規則 4: 貸款金額過高
    # 根據訓練數據，貸款金額超過年收入 5 倍的客戶核准率較低
    if loan_to_income > LOAN_TO_INCOME_MAX:
        final_prob = min(ai_prob, 0.3)
        reason = f"貸款金額過高（為年收入的 {loan_to_income:.1f} 倍 > {LOAN_TO_INCOME_MAX}）"
        if final_prob != original_prob:
            rule_triggered = True
        return final_prob, reason, rule_triggered
    
    # 規則 5: 中等風險客戶
    # DTI 在可接受範圍且信用分數良好
    if DTI_EXCELLENT <= dti <= DTI_ACCEPTABLE and credit_score >= CREDIT_SCORE_GOOD:
        # AI 模型決定，但給予正向調整
        final_prob = min(ai_prob * 1.1, 1.0)  # 小幅提升 10%
        if final_prob >= optimal_threshold:
            reason = f"條件良好（信用分數 {credit_score} ≥ {CREDIT_SCORE_GOOD}，負債比 {dti:.1%} 可接受）"
        else:
            reason = f"AI 模型評估（信用分數 {credit_score}，負債比 {dti:.1%}）"
        if final_prob != original_prob:
            rule_triggered = True
        return final_prob, reason, rule_triggered
    
    # 預設：使用 AI 模型結果
    if ai_prob >= optimal_threshold:
        reason = f"AI 模型建議核准（機率 {ai_prob:.1%} ≥ {optimal_threshold:.1%}）"
    else:
        reason = f"AI 模型建議婉拒（機率 {ai_prob:.1%} < {optimal_threshold:.1%}）"
    
    return ai_prob, reason, rule_triggered


def calculate_monthly_payment(principal, annual_rate, months):
    """
    計算月付金（使用標準貸款公式）
    
    參數:
        principal: 本金
        annual_rate: 年利率（百分比，例如 5.5）
        months: 還款月數
    
    返回:
        月付金
    """
    if principal <= 0 or months <= 0:
        return 0
    
    if annual_rate <= 0:
        # 無利率情況
        return principal / months
    
    # 月利率
    monthly_rate = annual_rate / 100 / 12
    
    # 使用標準房貸公式
    # M = P * [r(1+r)^n] / [(1+r)^n - 1]
    power = (1 + monthly_rate) ** months
    monthly_payment = principal * (monthly_rate * power) / (power - 1)
    
    return monthly_payment


def get_improvement_suggestions(dti, credit_score, mon_expense, mon_income, 
                                loan_amt, ann_income, dependents):
    """
    根據用戶財務狀況提供改善建議
    
    返回:
        建議列表
    """
    suggestions = []
    
    # 建議 1: 降低負債比
    if dti > DTI_ACCEPTABLE:
        suggestions.append({
            'icon': '💳',
            'title': '降低負債比',
            'content': f'您的負債比為 {dti:.1%}，建議降至 {DTI_ACCEPTABLE:.1%} 以下',
            'actions': [
                '優先償還高利率債務',
                '考慮債務整合',
                '增加收入來源（兼職、副業）'
            ]
        })
    
    # 建議 2: 改善信用評分
    if credit_score < CREDIT_SCORE_GOOD:
        suggestions.append({
            'icon': '⭐',
            'title': '提升信用評分',
            'content': f'您的信用評分為 {credit_score}，建議提升至 {CREDIT_SCORE_GOOD} 以上',
            'actions': [
                '準時繳納所有帳單（信用卡、水電費等）',
                '不要使用超過信用額度的 30%',
                '避免頻繁申請新的信用卡或貸款',
                '定期檢查信用報告，確保無誤'
            ]
        })
    
    # 建議 3: 控制支出
    expense_ratio = mon_expense / mon_income if mon_income > 0 else 0
    if expense_ratio > 0.6:
        suggestions.append({
            'icon': '💰',
            'title': '控制每月支出',
            'content': f'支出佔收入的 {expense_ratio:.1%}，建議控制在 60% 以下',
            'actions': [
                '檢視並削減非必要開支',
                '使用記帳 App 追蹤支出',
                '建立每月預算計畫',
                '考慮共享經濟（共乘、共住等）'
            ]
        })
    
    # 建議 4: 降低申請金額
    if loan_amt > ann_income * LOAN_TO_INCOME_MAX:
        recommended_amount = ann_income * LOAN_TO_INCOME_MAX
        suggestions.append({
            'icon': '📉',
            'title': '調整貸款金額',
            'content': f'建議將貸款金額降至年收入的 {LOAN_TO_INCOME_MAX} 倍以內',
            'actions': [
                f'建議申請金額: NT$ {recommended_amount:,.0f}',
                '或者增加頭期款比例',
                '考慮延長還款期限'
            ]
        })
    
    # 建議 5: 考慮共同借款人
    if dependents >= 3:
        suggestions.append({
            'icon': '👥',
            'title': '考慮共同借款人',
            'content': '扶養人數較多，建議加入配偶或家人作為共同借款人',
            'actions': [
                '與配偶或家人討論共同申請',
                '分散還款壓力',
                '可能獲得更好的貸款條件'
            ]
        })
    
    # 建議 6: 增加資產
    if not suggestions:
        suggestions.append({
            'icon': '📈',
            'title': '持續累積資產',
            'content': '您的財務狀況良好，建議持續累積資產',
            'actions': [
                '定期定額儲蓄或投資',
                '建立緊急備用金（3-6個月生活費）',
                '考慮購置保值資產',
                '6-12 個月後可重新評估申請更優惠條件'
            ]
        })
    
    return suggestions


def estimate_assets(ann_income, self_employed):
    """
    基於收入估算資產（用於預測時沒有實際資產數據的情況）
    
    參數:
        ann_income: 年收入
        self_employed: 就業類型 (" No" 或 " Yes")
    
    返回:
        (住宅資產, 商業資產, 奢侈品資產, 銀行存款, 總資產)
    """
    est_residential = ann_income * ASSET_RESIDENTIAL_MULTIPLIER
    
    if self_employed == " Yes":
        est_commercial = ann_income * ASSET_COMMERCIAL_SELFEMPLOYED
    else:
        est_commercial = ann_income * ASSET_COMMERCIAL_EMPLOYED
    
    est_luxury = ann_income * ASSET_LUXURY_MULTIPLIER
    est_bank = ann_income * ASSET_BANK_MULTIPLIER
    
    total_assets = est_residential + est_commercial + est_luxury + est_bank
    
    return est_residential, est_commercial, est_luxury, est_bank, total_assets
