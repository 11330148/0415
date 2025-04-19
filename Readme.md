import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 2. 前處理：新增每坪價格欄位
df["每坪價格"] = df["總價"] / df["建物總面積"]

# 3. 選擇特徵（不包含總價與每坪價格）
features = ["土地面積", "建物總面積", "屋齡", "樓層", "總樓層", "房數", "廳數", "衛數", "電梯"]
# 將「電梯」欄轉換為數值（0, 1）
df["電梯"] = df["電梯"].map({0: 0, 1: 1, "有": 1, "無": 0})

# 去除缺失值與非數值行
df = df.dropna(subset=features + ["總價"])

# 4. 分割資料
X = df[features]
y = df["總價"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 建立模型並訓練
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 6. 預測與評估
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"模型 RMSE：{rmse:.2f} 萬元")

# 7. 顯示特徵重要性
importances = model.feature_importances_
for name, importance in zip(features, importances):
    print(f"{name}: {importance:.3f}")
