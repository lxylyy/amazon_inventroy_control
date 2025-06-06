import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Inventory Optimization Dynamic Programming", layout="wide")
st.title("📦 Walmart Inventory Dynamic Programming Model (Supports Holidays & Cost Adjustment)")

# 上传数据
uploaded_files = st.file_uploader("Upload two CSV files (features.csv, stores.csv)", accept_multiple_files=True)
if uploaded_files and len(uploaded_files) == 2:
    uploaded_file = uploaded_files[0]
    uploaded_file2 = uploaded_files[1]

    df = pd.read_csv(uploaded_file)
    df = df[df['Store'] == 1]  # 仅使用 Store 1 的数据

    store_df = pd.read_csv(uploaded_file2)
    store_df['Size_Factor'] = store_df['Size'] / store_df['Size'].mean()
    store_df = store_df[store_df['Store'] == 1]  # 仅使用 Store 1 的数据

    # merge two dataframes on 'Store' column
    df = df.merge(store_df[['Store', 'Type', 'Size', 'Distance_km', 'Size_Factor']], on='Store', how='left')

    # 模拟 Demand：气温敏感商品（如冰饮料）
    np.random.seed(42)
    base_demand = np.random.uniform(80, 120, len(df))
    df['Demand'] = base_demand * df['Size_Factor'] * (df['CPI'] / df['CPI'].mean())

    # 节假日 ×1.5 放大
    df['Demand'] = df['Demand'].where(~df['IsHoliday'], df['Demand'] * 1.5)

    # 气温影响：气温越高，需求越高（每升高1°C，需求上涨3%，基于20°C）
    df['Demand'] *= 1 + 0.03 * (df['Temperature'] - 20)

    # 模拟 Unit_Cost：固定成本 + 随机扰动 + 假期促销
    np.random.seed(42)
    df['Unit_Cost'] = np.random.normal(loc=6.5, scale=0.2, size=len(df))
    df.loc[df['IsHoliday'], 'Unit_Cost'] *= 0.9  # 假期促销


    # 保留关键列
    # df = df[['Date', 'Demand', 'Unit_Cost', 'IsHoliday', 'Temperature']]
    df['Date'] = pd.to_datetime(df['Date'])


    # 参数设置
    hold_ratio = 0.2
    shortage_multiplier = 4
    st.sidebar.header("Model Parameter Settings")
    initial_inventory = st.sidebar.number_input("Initial Inventory Level I₀ (0-1000)", min_value=0, max_value=1000, value=50)
    max_order = st.sidebar.number_input("Max order quantity per period Qₜ (10-1000)", min_value=10, max_value=1000, value=100)

    # 初始化 DP
    T = len(df)
    inventory_levels = range(0, 210, 10)
    dp = {}
    policy = {}
    cost_sum = 0
    
    # 倒推法动态规划
    for t in reversed(range(T)):
        dp[t] = {}
        policy[t] = {}
        demand = df.loc[t, 'Demand']
        cost = df.loc[t, 'Unit_Cost']
        is_holiday = df.loc[t, 'IsHoliday']
        if is_holiday:
            demand *= 1.5  # 节假日放大

        # 气温越高，冷藏成本越高（25°C为冷藏临界点）
        h = hold_ratio * cost * (1 + 0.02 * max(0, df.loc[t, 'Temperature'] - 25))

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
            cost_sum += total_cost

    # 输出最优进货策略路径
    st.subheader("📊 Recommended Ordering Policy")
    inventory = initial_inventory
    plan = []
    for t in range(T):
        q = policy[t][int(round(inventory / 10) * 10)]
        demand = df.loc[t, 'Demand']
        plan.append({
            "Date": (df.loc[t, 'Date'] + pd.DateOffset(years=15)).strftime("%Y-%m-%d"),
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

    # 计算total cost并显示
    st.metric(label="Total Cost", value=f"${cost_sum:,.2f}")
