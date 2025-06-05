import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Inventory Optimization Dynamic Programming", layout="wide")
st.title("📦 Walmart Inventory Dynamic Programming Model (Supports Holidays & Cost Adjustment)")

# 上传数据
uploaded_file = st.file_uploader("Upload CSV data containing Demand, Unit_Cost, IsHoliday", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 校验基础列是否存在
    required_cols = {'Date', 'Fuel_Price', 'CPI', 'IsHoliday'}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ Uploaded file is missing required columns: {required_cols - set(df.columns)}")
        st.stop()

    # 模拟 Demand：基础随机 × CPI × 节假日放大
    np.random.seed(42)
    base_demand = np.random.uniform(80, 120, len(df))
    df['Demand'] = base_demand * (df['CPI'] / df['CPI'].mean())
    df['Demand'] = df['Demand'].where(~df['IsHoliday'], df['Demand'] * 1.5)  # 节假日加倍

    # 模拟 Unit_Cost：与 CPI 和油价相关联
    df['Unit_Cost'] = 6.5 + 0.02 * (df['CPI'] - df['CPI'].mean()) + 0.2 * (df['Fuel_Price'] - df['Fuel_Price'].mean())

    # 保留关键列
    df = df[['Date', 'Demand', 'Unit_Cost', 'IsHoliday']]
    df['Date'] = pd.to_datetime(df['Date'])

    # 参数设置
    st.sidebar.header("Model Parameter Settings")
    hold_ratio = st.sidebar.slider("Holding cost ratio (h = ratio × purchase cost)", 0.05, 0.5, 0.1)
    shortage_multiplier = st.sidebar.slider("Shortage cost multiplier (p = multiplier × purchase cost)", 1.0, 5.0, 2.0)
    initial_inventory = st.sidebar.number_input("Initial Inventory Level I₀", min_value=0, max_value=1000, value=50)
    max_order = st.sidebar.number_input("Max order quantity per period Qₜ", min_value=10, max_value=1000, value=100)

    # 初始化 DP
    T = len(df)
    inventory_levels = range(0, 210, 10)
    dp = {}
    policy = {}
    
    # 倒推法动态规划
    for t in reversed(range(T)):
        dp[t] = {}
        policy[t] = {}
        demand = df.loc[t, 'Demand']
        cost = df.loc[t, 'Unit_Cost']
        is_holiday = df.loc[t, 'IsHoliday']
        if is_holiday:
            demand *= 1.5  # 节假日放大

        h = hold_ratio * cost
        p = shortage_multiplier * cost

        for inv in inventory_levels:
            min_cost = float('inf')
            best_q = 0
            for q in range(0, max_order + 1, 10):
                next_inv = max(0, inv + q - demand)
                shortage = max(0, demand - (inv + q))
                immediate_cost = cost * q + h * next_inv + p * shortage
                future_cost = dp[t + 1].get(next_inv, 0) if t + 1 in dp else 0
                total_cost = immediate_cost + future_cost
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_q = q
            dp[t][inv] = min_cost
            policy[t][inv] = best_q

    # 输出最优进货策略路径
    st.subheader("📊 推荐进货策略")
    inventory = initial_inventory
    plan = []
    for t in range(T):
        q = policy[t][int(round(inventory / 10) * 10)]
        demand = df.loc[t, 'Demand']
        plan.append({
            "Date": df.loc[t, 'Date'].strftime("%Y-%m-%d"),
            "Inventory_Begin": inventory,
            "Order_Q": q,
            "Demand": round(demand, 1),
            "Inventory_End": max(0, inventory + q - demand)
        })
        inventory = max(0, inventory + q - demand)

    result_df = pd.DataFrame(plan)
    st.dataframe(result_df)

    # 成本趋势图
    st.subheader("📈 Order Quantity per Period Visualization")
    st.bar_chart(result_df.set_index("Date")["Order_Q"])
