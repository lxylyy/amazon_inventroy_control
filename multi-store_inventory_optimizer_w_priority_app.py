import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="Warehouse to Multi-Store Inventory Optimization", layout="wide")
st.title("🏪 Walmart Cold Beverage: Central Warehouse → Multi-Store DP & Allocation Optimization")

# ========== 1. 上传数据 ========== #
features_file = st.file_uploader("Upload Walmart features.csv (with Date, Fuel_Price, CPI, Temperature, IsHoliday)", type="csv")
if not features_file:
    st.stop()
df = pd.read_csv(features_file)
df = df[df['Store'].isin(range(1, 6))]  # 仅使用 Store 1-5 的数据
df['Date'] = pd.to_datetime(df['Date'])

# ========== 2. 门店信息 & 距离（真实模拟）========== #
store_df = pd.DataFrame({
    'Store': [1, 2, 3, 4, 5],
    'Type': ['A', 'A', 'B', 'A', 'B'],
    'Size': [151315, 202307, 37392, 205863, 34875],
    'Distance_km': [18.6, 49.5, 25.2, 12.3, 34.8],
})
store_df['Max_Inventory'] = (store_df['Size'] / 200).astype(int)
store_df['Size_Factor'] = store_df['Size'] / store_df['Size'].mean()

# ========== 3. 需求生成 ========== #
np.random.seed(42)
T = len(df)
S = len(store_df)
base_demand = np.random.uniform(80, 120, (S, T))
demand_matrix = []
for i, row in store_df.iterrows():
    demand = base_demand[i] * row['Size_Factor']
    demand *= (df['CPI'] / df['CPI'].mean()).values
    demand *= (1 + 0.03 * (df['Temperature'] - 20)).values
    demand = np.where(df['IsHoliday'], demand * 1.5, demand)
    demand_matrix.append(demand)
demand_matrix = np.array(demand_matrix)

# ========== 4. 配送成本矩阵 ========== #
fuel_price = df['Fuel_Price'].mean()
distance_matrix = store_df['Distance_km'].values.reshape(-1, 1)
transport_cost_matrix = 0.05 * distance_matrix * fuel_price  # 元/件

# ========== 5. 参数设置 ========== #
st.sidebar.header("Global Parameters")
initial_inventory = st.sidebar.number_input("Initial Warehouse Inventory", min_value=0, value=3000)
max_order = st.sidebar.slider("Max Order to Warehouse per Period", 100, 5000, 1500)
hold_ratio = 0.2
shortage_multiplier = 4

# ========== 6. 中央仓订购动态规划 ========== #
warehouse_plan = []
warehouse_inventory = initial_inventory
warehouse_orders = []
store_inventory = np.zeros(S)
store_plan = []

# 优先级权重（Type A > B > C）
priority_weights = {'A': 3, 'B': 2, 'C': 1}
store_df['Priority'] = store_df['Type'].map(priority_weights)

for t in range(T):
    # ====== 仓库补货决策 ======
    future_demand = demand_matrix[:, t:].sum(axis=1).sum() / (T - t)
    reorder_qty = max_order if warehouse_inventory < future_demand / 4 else 0
    warehouse_orders.append(reorder_qty)
    warehouse_inventory += reorder_qty

    # ====== 本期成本参数 ======
    unit_cost = 6.5
    temp = df.loc[t, 'Temperature']
    hold_cost = hold_ratio * unit_cost * (1 + 0.02 * max(0, temp - 25))
    shortage_cost = shortage_multiplier * unit_cost
    demands = demand_matrix[:, t]
    max_inv = store_df['Max_Inventory'].values
    supply_limit = np.minimum(np.maximum(0, max_inv - store_inventory), demands)
    supply_limit = np.minimum(supply_limit, warehouse_inventory)

    # 优先级惩罚项加入成本：优先级越高，惩罚越低
    priority_cost = (store_df['Priority'].max() - store_df['Priority'].values) * 1.0  # 每优先级差1增加1元成本
    c = hold_cost * np.ones(S) + transport_cost_matrix.flatten() + priority_cost

    A_eq = [np.ones(S)]
    b_eq = [min(max_order, warehouse_inventory)]
    bounds = [(0, float(s)) for s in supply_limit]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    allocation = res.x if res.success else np.zeros(S)

    # 更新库存
    warehouse_inventory -= allocation.sum()
    store_inventory += allocation - demands
    store_inventory = np.maximum(store_inventory, 0)

    for i, store_id in enumerate(store_df['Store']):
        store_plan.append({
            "Date": df.loc[t, 'Date'].strftime("%Y-%m-%d"),
            "Store": store_id,
            "Demand": round(demands[i], 1),
            "Allocated": round(allocation[i], 1),
            "Inv_After": round(store_inventory[i], 1),
            "Hold_Cost": round(hold_cost * store_inventory[i], 2),
            "Transport_Cost": round(transport_cost_matrix[i][0] * allocation[i], 2),
            "Priority": store_df.loc[i, 'Priority']
        })

    warehouse_plan.append({
        "Date": df.loc[t, 'Date'].strftime("%Y-%m-%d"),
        "Warehouse_Inventory": warehouse_inventory,
        "Order_Qty": reorder_qty
    })

# ========== 7. 可视化结果 ========== #
st.subheader("📦 Central Warehouse Order Plan")
wh_df = pd.DataFrame(warehouse_plan)
st.dataframe(wh_df)
st.bar_chart(wh_df.set_index("Date")["Order_Qty"])

st.subheader("📊 Multi-Store Inventory Allocation Plan")
result_df = pd.DataFrame(store_plan)
st.dataframe(result_df)

st.subheader("📈 Per-Store Allocation Over Time")
pivot_chart = result_df.pivot(index="Date", columns="Store", values="Allocated")
st.line_chart(pivot_chart)