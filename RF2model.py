# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:42:36 2024

@author: kangjia
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import layers
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization
import seaborn as sns
import os
from datetime import datetime,timedelta
import re
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib import colors   
from scipy import stats   
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import joblib


# In[样本集前处理] 
landslides = pd.read_excel('E:/LSM/有监督/landslide_train.xlsx')
nolandslides = pd.read_excel('E:/LSM/有监督/nolandslides_RF.xls')

landslides = landslides.replace(-9999, np.nan)
landslides=landslides.dropna(subset=(landslides.columns))#.astype(np.float32)

nolandslides = nolandslides.replace(-9999, np.nan)
nolandslides=nolandslides.dropna(subset=(nolandslides.columns))#.astype(np.float32)



column_need = ['DEM', 'slope', 'aspect', 'TWI','faults_den', 'landcover', 'Lithology', 'NDVI', 'rainfall', 'river_den',
       'road_den', 'soil']
landslides_X = landslides[column_need]
nolandslides_X = nolandslides[column_need]

landslides['label'] = 1
nolandslides['label'] = 0

Xtrain0, Xtest0, Ytrain0, Ytest0 = train_test_split(nolandslides_X,nolandslides['label'],test_size=0.3,random_state=1000)#分训练集和测试集

Xtrain = pd.concat([landslides_X,Xtrain0],0)
Ytrain = pd.concat([landslides['label'],Ytrain0],0)

test1 = pd.read_excel('E:/LSM/有监督/Xtest1.xlsx')
test1['label'] = 1
Xtest = pd.concat([test1[column_need],Xtest0.sample(len(test1))],0)
Ytest = pd.concat([test1['label'],Ytest0.sample(len(test1))],0)


# In[]训练模型


# rf训练
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=int(max_features), # float
            max_depth=int(max_depth),
            random_state=2),Xtrain, Ytrain.ravel(), scoring='accuracy', cv=5).mean()
    return val
rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (10, 200),
        'min_samples_split': (2,10),
        'max_features': (2, 20),
        'max_depth': (2, 50)})
rf_bo.maximize()


index = []
for i in rf_bo.res:
    index.append(i['target'])
max_index = index.index(max(index))
rf_bo.res[max_index]


best_params = { k: int (v) for k, v in rf_bo.res[max_index]['params'].items ()}

rf = RandomForestClassifier(**best_params)


#计算一些指标
rf.fit(Xtrain, Ytrain.ravel())
Ytest_predict = rf.predict(Xtest)
Ytrain_predict = rf.predict(Xtrain)


# 计算分类报告
from sklearn.metrics import classification_report, confusion_matrix
class_report = classification_report(Ytest, Ytest_predict,digits=3)
print("分类报告：")
print(class_report)

# In[]保存模型

#保存模型到文件 
joblib.dump(rf, 'E:/LSM/有监督/rf2.pkl')

# 加载模型 loaded_model = joblib.load('random_forest_model.pkl')


# In[]

#预测整个区域易发性图
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as plt
import os
import rioxarray as rxr
import rasterio
import joblib

tif_path = 'E:/LSM/factors/'
column_need = ['DEM', 'slope', 'aspect', 'TWI','faults_den', 'landcover', 'Lithology', 'NDVI', 'rainfall', 'river_den',
       'road_den', 'soil']
df = pd.DataFrame()
for name in column_need:
    tif_file = tif_path + name + '.tif'
    temp = rxr.open_rasterio(tif_file)
    temp = temp.where(temp != temp._FillValue, np.nan)
    df[name] = temp.data.flatten()
lon_x, lat_y = temp.x, temp.y   


df_data = df.dropna(subset=(df.columns))#.astype(np.float32)


print(df_data.isnull().sum())
X_result = df_data

rf = joblib.load('E:/LSM/有监督/rf2.pkl')
#Y_result = rf.predict(X_result)
# 获取测试集样本的连续概率值
Y_result_proba = rf.predict_proba(X_result)

df_data['label'] = Y_result_proba[:,1]

df['label'] = df_data['label']

label = np.array(df['label']).reshape(1,3880, 4160)

df_nc = temp
df_nc.data = label

with rasterio.Env():
    df_nc.rio.to_raster('E:/LSM/有监督/RF2_LSI.tif')
df_nc.plot()





# In[] import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

rf = joblib.load('E:/LSM/有监督/rf2.pkl')
# 假设 y_true 是实际标签，y_score 是预测的概率分数
y_true = Ytest
y_score = rf.predict_proba(Xtest)[:,1]

# 计算ROC曲线的数据
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# 将ROC数据保存到CSV文件中
roc_data = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr,
    'Thresholds': thresholds
})
roc_data.to_csv('E:/LSM/有监督/rf2_roc_data.csv', index=False)

print("ROC数据已保存到roc_data.csv文件中")

# 绘制ROC曲线的代码
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(fpr, tpr, roc_auc)









#绘制重要性图
def plot_features_weigLEs(model, weigLEs, feature_names, weigLEs_type='c'):
    
    """Plot regression coefficients weigLEs or feature importance.
    
    Args:
        model (str): The model name identifier
        weigLEs (array): Contains the regression coefficients weigLEs or feature importance
        feature_names (list): Contains the corresponding features names
        weigLEs_type (str): 'c' for 'coefficients weigLEs', otherwise is 'feature importance'
        
    Returns:
        plot of either regression coefficients weigLEs or feature importance
        
    
    """
    plt.rcParams['savefig.dpi'] = 300 #图片像素(根据需求调,值太小图看不清)
    plt.rcParams['figure.dpi'] = 300 
    (px, py) = (8, 10) if len(weigLEs) > 30 else (8, 5)
    W = pd.DataFrame({'WeigLEs':weigLEs}, feature_names)
    W.sort_values(by='WeigLEs', ascending=True).plot(kind='barh', color='b', figsize=(px,py))
    label = ' Coefficients' if weigLEs_type =='c' else ' Features Importance'
    plt.xlabel(model + label)
    plt.gca().legend_ = None
    



plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.size'] = '15'
plot_features_weigLEs('Random Forest Regressor', rf.feature_importances_, Xtrain.columns, 't' )#特征重要性排序







