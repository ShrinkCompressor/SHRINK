import numpy as np
import xgboost as xgb

# 生成随机时序数据
data = np.random.uniform(-2, 2, 1000)

# 创建滞后特征
def create_lagged_features(data, n_lags=5):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_lagged_features(data)

# 训练 XGBoost 模型
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X, y)

# 进行预测并计算残差
predictions = model.predict(X)
residuals = y - predictions

# 计算压缩比和误差
compression_ratio = len(data) / len(residuals)
error_percentage = np.mean(np.abs(residuals / y)) * 100

print("Compression Ratio:", compression_ratio)
print("Error Percentage:", error_percentage)
