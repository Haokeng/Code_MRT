import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
sns.set_theme(style="ticks", color_codes=True)
from tqdm.notebook import tqdm, trange
import time
start_time = time.time()

#讀入檔案
main_2017 = pd.read_csv('2017.csv')
main_2018 = pd.read_csv('2018.csv')
main_2019 = pd.read_csv('2019.csv')
main_2020 = pd.read_csv('2020.csv')
main_2021 = pd.read_csv('2021.csv')
main_2022 = pd.read_csv('2022.csv')

end_time = time.time()
total_time = end_time - start_time

print("讀入全部資料集,所花運行時間：", total_time, "秒")

start_time = time.time()
full_year_data = pd.concat([main_2017,main_2018, main_2019, main_2020, main_2021, main_2022])
end_time = time.time()
total_time = end_time - start_time

print("整合六年資料集,所花運行時間：", total_time, "秒")


#繪製六年資料圖譜

start_time = time.time()

ppl_every_day_all = full_year_data.groupby(['Date']).sum().reset_index().drop(['Hour'],axis=1)
ppl_every_day_all

import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_axes([0, 0, 1, 1])

# 將 Date 欄位轉換為日期格式
ppl_every_day_all['Date'] = pd.to_datetime(ppl_every_day_all['Date'])

# 建立顏色映射
color_map = {2017: 'red', 2018: 'orange', 2019: 'blue', 2020: 'green', 2021: 'purple', 2022: 'brown'}

# 逐年繪製線條
for year, color in color_map.items():
    year_data = ppl_every_day_all[ppl_every_day_all['Date'].dt.year == year]
    ax1.plot(year_data['Date'], year_data['CrowdFlow'], color=color)

plt.show()

end_time = time.time()
total_time = end_time - start_time

print("輸出六年圖表,所花運行時間：", total_time, "秒")


#轉換資料表格,插入欄位
start_time = time.time()

full_year_data['Date'] = pd.to_datetime(full_year_data['Date'])
full_year_data['DayofWeek'] = full_year_data['Date'].dt.dayofweek + 1
full_year_data['Month'] = full_year_data['Date'].dt.month
full_year_data

tidy_full_data = full_year_data
tidy_full_data

end_time = time.time()
total_time = end_time - start_time

print("轉換資料表格,插入欄位,所花運行時間：", total_time, "秒")

#六年資料切分
start_time = time.time()

train_size = int(len(tidy_full_data) * 0.6)
validation_size = int(len(tidy_full_data) * 0.2)
test_size = int(len(tidy_full_data) * 0.2)

train_data = tidy_full_data[0:train_size]
val_data = tidy_full_data[train_size:(train_size+validation_size)]
test_data = tidy_full_data[(train_size+validation_size):len(tidy_full_data)]

end_time = time.time()
total_time = end_time - start_time

print("六年資料切分,形成訓練集,驗證集,測試集,所花運行時間：", total_time, "秒")


#轉換車站代號,以平均數轉換
start_time = time.time()

ppl_per_station = train_data.drop(['Date'],axis=1).groupby(['Station']).sum().reset_index().drop(['Hour'],axis=1)
station_ranking = ppl_per_station.sort_values(by=['CrowdFlow'],ascending=False)
station_ranking = station_ranking.drop(['DayofWeek','Month'],axis=1).reset_index().drop(['index'],axis=1)

di = station_ranking.set_index('Station').to_dict()['CrowdFlow']

for k, v in di.items():
    di[k] = round(v/439)

end_time = time.time()
total_time = end_time - start_time

print("轉換車站代號,以平均數轉換,所花運行時間：", total_time, "秒")

#轉換代號
start_time = time.time()
train_data = train_data.replace({"Station": di})
train_data = train_data.drop(['Date'],axis=1)

test_data = test_data.replace({"Station": di})
test_data = test_data.drop(['Date'],axis=1)

val_data = val_data.replace({"Station": di})
val_data = val_data.drop(['Date'],axis=1)

end_time = time.time()
total_time = end_time - start_time

print("資料表轉換車站代號,所花運行時間：", total_time, "秒")

#分割資料集
start_time = time.time()
X_train = train_data.drop(['CrowdFlow'],axis=1)
y_train = train_data['CrowdFlow']
X_test = test_data.drop(['CrowdFlow'],axis=1)
y_test = test_data['CrowdFlow']
X_val = val_data.drop(['CrowdFlow'],axis=1)
y_val = val_data['CrowdFlow']

end_time = time.time()
total_time = end_time - start_time

print("形成XY資料集,所花運行時間：", total_time, "秒")

#運算六年隨機森林模型
start_time = time.time()
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_val = np.array(X_val)
y_val = np.array(y_val)

from sklearn.ensemble import RandomForestRegressor

rforest = RandomForestRegressor(n_estimators = 100, min_samples_split=10,random_state = 0)
rforest.fit(X_train, y_train)

end_time = time.time()
total_time = end_time - start_time

print("運算六年隨機森林模型,所花運行時間：", total_time, "秒")

#計算R2分數
start_time = time.time()
#使用X_val測試集預測predictions
predictions = rforest.predict(X_val)

#R Square
import statsmodels.api as sm
X_addC = sm.add_constant(predictions)
result = sm.OLS(y_val, X_addC).fit()
print(result.rsquared, result.rsquared_adj)
#RMSE
from sklearn.metrics import mean_squared_error
import math
#First model
print(mean_squared_error(y_val, predictions))
print(math.sqrt(mean_squared_error(y_val, predictions)))

true_data = pd.DataFrame(data = y_val)

predictions_data = pd.DataFrame(data = predictions)

combined = predictions_data
combined['Actual'] = true_data[0]
combined.rename(columns = {0: 'Predicted'}, inplace = True)
random_combined = combined.sample(n=250, random_state=1)
"""
%matplotlib inline

f, ax = plt.subplots(figsize=(20, 40))

# Plot the actual values
plt.plot(true_data[1000:1250], 'b-', label='actual')

# Plot the predicted values
plt.plot(predictions_data[1000:1250]['Predicted'], 'go', label='prediction (Predicted)')
plt.plot(predictions_data[1000:1250]['Actual'], 'ro', label='prediction (Actual)')

plt.xticks(rotation='60')
plt.legend()

plt.xlabel('Samples')
plt.ylabel('Traffic')
plt.title('Actual and Predicted Values')
"""
end_time = time.time()
total_time = end_time - start_time

print("資料表轉換車站代號,所花運行時間：", total_time, "秒")


#切
start_time = time.time()
import pandas as pd

# 假設 tidy_full_data 是一個 pandas DataFrame，且有 'Date' 欄位
start_date = '2017-01-01'
end_date = '2019-12-31'

# 將 'Date' 欄位轉換為日期型態
tidy_full_data['Date'] = pd.to_datetime(tidy_full_data['Date'])

# 篩選 'Date' 在指定日期範圍內的資料欄位
filtered_data = tidy_full_data[(tidy_full_data['Date'] >= start_date) & (tidy_full_data['Date'] <= end_date)]

end_time = time.time()
total_time = end_time - start_time

print("2017~2019疫情前資料切分：", total_time, "秒")

###訓練機切分
start_time = time.time()
train_size_before = int(len(filtered_data) * 0.6)
validation_size_before = int(len(filtered_data) * 0.2)
test_size_before = int(len(filtered_data) * 0.2)

train_data_before = filtered_data[0:train_size_before]
val_data_before = filtered_data[train_size_before:(train_size_before+validation_size_before)]
test_data_before = filtered_data[(train_size_before+validation_size_before):len(filtered_data)]
end_time = time.time()
total_time = end_time - start_time

print("疫情前資料切分資料集：", total_time, "秒")

#轉換代碼
start_time = time.time()
train_data_before = train_data_before.replace({"Station": di})
train_data_before = train_data_before.drop(['Date'],axis=1)

test_data_before = test_data_before.replace({"Station": di})
test_data_before = test_data_before.drop(['Date'],axis=1)

val_data_before = val_data_before.replace({"Station": di})
val_data_before = val_data_before.drop(['Date'],axis=1)
end_time = time.time()
total_time = end_time - start_time

print("疫情前資料轉換代碼：", total_time, "秒")

#
start_time = time.time()
X_train_before = train_data_before.drop(['CrowdFlow'],axis=1)
y_train_before = train_data_before['CrowdFlow']
X_test_before = test_data_before.drop(['CrowdFlow'],axis=1)
y_test_before = test_data_before['CrowdFlow']
X_val_before = val_data_before.drop(['CrowdFlow'],axis=1)
y_val_before = val_data_before['CrowdFlow']
end_time = time.time()
total_time = end_time - start_time

print("疫情前資料切分XY：", total_time, "秒")

#2017~2019模型
start_time = time.time()
X_train_before = np.array(X_train_before)
X_test_before = np.array(X_test_before)
y_train_before = np.array(y_train_before)
y_test_before = np.array(y_test_before)
X_val_before = np.array(X_val_before)
y_val_before = np.array(y_val_before)

from sklearn.ensemble import RandomForestRegressor

rforest_before = RandomForestRegressor(n_estimators = 100, min_samples_split=10,random_state = 0)
rforest_before.fit(X_train_before, y_train_before)
end_time = time.time()
total_time = end_time - start_time

print("疫情前模型訓練：", total_time, "秒")

#分數計算
start_time = time.time()
predictions_before = rforest_before.predict(X_val_before)

#R Square
import statsmodels.api as sm
X_addC_before = sm.add_constant(predictions_before)
result_before = sm.OLS(y_val_before, X_addC_before).fit()
print(result_before.rsquared, result_before.rsquared_adj)
#RMSE
from sklearn.metrics import mean_squared_error
import math
#First model
print(mean_squared_error(y_val_before, predictions_before))
print(math.sqrt(mean_squared_error(y_val_before, predictions_before)))

true_data_before = pd.DataFrame(data = y_val_before)

predictions_data_before = pd.DataFrame(data = predictions_before)

combined_before = predictions_data_before
combined_before['Actual'] = true_data_before[0]
combined_before.rename(columns = {0: 'Predicted'}, inplace = True)
random_combined_before = combined_before.sample(n=250, random_state=1)

"""
%matplotlib inline

f, ax = plt.subplots(figsize=(20, 40))

# Plot the actual values
plt.plot(true_data_before[1000:1250], 'b-', label='actual')

# Plot the predicted values
plt.plot(predictions_data_before[1000:1250]['Predicted'], 'go', label='prediction (Predicted)')
plt.plot(predictions_data_before[1000:1250]['Actual'], 'ro', label='prediction (Actual)')

plt.xticks(rotation='60')
plt.legend()

plt.xlabel('Samples')
plt.ylabel('Traffic')
plt.title('Actual and Predicted Values')
"""
end_time = time.time()
total_time = end_time - start_time

print("运行时间：", total_time, "秒")

####
import pandas as pd

# 假設 tidy_full_data 是一個 pandas DataFrame，且有 'Date' 欄位
start_date = '2017-01-01'
end_date = '2019-12-31'

# 將 'Date' 欄位轉換為日期型態
tidy_full_data['Date'] = pd.to_datetime(tidy_full_data['Date'])

# 篩選 'Date' 在指定日期範圍內的資料欄位
filtered_data = tidy_full_data[(tidy_full_data['Date'] >= start_date) & (tidy_full_data['Date'] <= end_date)]


train_size_before = int(len(filtered_data) * 0.6)
validation_size_before = int(len(filtered_data) * 0.2)
test_size_before = int(len(filtered_data) * 0.2)

train_data_before = filtered_data[0:train_size_before]
val_data_before = filtered_data[train_size_before:(train_size_before+validation_size_before)]
test_data_before = filtered_data[(train_size_before+validation_size_before):len(filtered_data)]

train_data_before = train_data_before.replace({"Station": di})
train_data_before = train_data_before.drop(['Date'],axis=1)

test_data_before = test_data_before.replace({"Station": di})
test_data_before = test_data_before.drop(['Date'],axis=1)

val_data_before = val_data_before.replace({"Station": di})
val_data_before = val_data_before.drop(['Date'],axis=1)

X_train_before = train_data_before.drop(['CrowdFlow'],axis=1)
y_train_before = train_data_before['CrowdFlow']
X_test_before = test_data_before.drop(['CrowdFlow'],axis=1)
y_test_before = test_data_before['CrowdFlow']
X_val_before = val_data_before.drop(['CrowdFlow'],axis=1)
y_val_before = val_data_before['CrowdFlow']

import xgboost as xgb
start_time = time.time()
# 创建模型
model = xgb.XGBRegressor()
# 训练模型
model.fit(X_train_before, y_train_before)
end_time = time.time()
total_time = end_time - start_time

print("xgboost运行时间：", total_time, "秒")

from sklearn.metrics import mean_squared_error
# 预测
y_pred_before = model.predict(X_test_before)

# 计算均方根误差（RMSE）
rmse = mean_squared_error(y_test_before, y_pred_before, squared=False)
print("模型的均方根误差（RMSE）：", rmse)

predictions_before =model.predict(X_val_before)

#R Square
import statsmodels.api as sm
X_addC_before = sm.add_constant(predictions_before)
result_before = sm.OLS(y_val_before, X_addC_before).fit()
print(result_before.rsquared, result_before.rsquared_adj)


#多層感知機
start_time = time.time()
import pandas as pd

# 假設 tidy_full_data 是一個 pandas DataFrame，且有 'Date' 欄位
start_date = '2017-01-01'
end_date = '2019-12-31'

# 將 'Date' 欄位轉換為日期型態
tidy_full_data['Date'] = pd.to_datetime(tidy_full_data['Date'])

# 篩選 'Date' 在指定日期範圍內的資料欄位
filtered_data = tidy_full_data[(tidy_full_data['Date'] >= start_date) & (tidy_full_data['Date'] <= end_date)]
filtered_data
#轉成單熱編碼確保為類型資料
encoded_df = pd.get_dummies(filtered_data, columns=["Hour", "Station", "DayofWeek", "Month"])

df_MLP = encoded_df

train_size_df_MLP = int(len(df_MLP) * 0.6)
validation_size_df_MLP = int(len(df_MLP) * 0.2)
test_size_df_MLP = int(len(df_MLP) * 0.2)

train_data_df_MLP = df_MLP[0:train_size_df_MLP]
val_data_df_MLP = df_MLP[train_size_df_MLP:(train_size_df_MLP+validation_size_df_MLP)]
test_data_df_MLP = df_MLP[(train_size_df_MLP+validation_size_df_MLP):len(df_MLP)]

X_train_df_MLP = train_data_df_MLP.drop(['CrowdFlow'],axis=1)
y_train_df_MLP = train_data_df_MLP['CrowdFlow']
X_test_df_MLP = test_data_df_MLP.drop(['CrowdFlow'],axis=1)
y_test_df_MLP = test_data_df_MLP['CrowdFlow']
X_val_df_MLP = val_data_df_MLP.drop(['CrowdFlow'],axis=1)
y_val_df_MLP = val_data_df_MLP['CrowdFlow']

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import sklearn
import os
sns.set()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

X_train_df_MLP = X_train_df_MLP.drop("Date", axis=1)
X_test_df_MLP = X_test_df_MLP.drop("Date", axis=1)
X_val_df_MLP = X_val_df_MLP.drop("Date", axis=1)

model = Sequential()    
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

# 定义模型
model.add(Dense(64, activation='relu', input_shape=(X_train_df_MLP.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1) )  # 输出层不使用激活函数

model.summary()

# 编译模型
model.compile(optimizer='adam', loss='mse',metrics=['mae'])

# 训练模型
model.fit(X_train_df_MLP, y_train_df_MLP, epochs=20, batch_size=32)

end_time = time.time()
total_time = end_time - start_time

print("深度模型運行時間：", total_time, "秒")
#############
start_time = time.time()
predictions =model.predict(X_val_df_MLP)

#R Square
import statsmodels.api as sm
X_addC = sm.add_constant(predictions)
result = sm.OLS(y_val_df_MLP, X_addC).fit()
print(result.rsquared, result.rsquared_adj)

#RMSE
from sklearn.metrics import mean_squared_error
import math
#First model
print(mean_squared_error(y_val_df_MLP, predictions))
print(math.sqrt(mean_squared_error(y_val_df_MLP, predictions)))

end_time = time.time()
total_time = end_time - start_time

print("深度模型運行時間：", total_time, "秒")