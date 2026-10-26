
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
landslides = pd.read_excel('E:/LSM/landslides/landslides.xls')
nolandslides = pd.read_excel('E:/LSM/landslides/nolandslides_random.xls')

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


Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(landslides_X,landslides['label'],test_size=0.3,random_state=1000)#分训练集和测试集
Xtrain0, Xtest0, Ytrain0, Ytest0 = train_test_split(nolandslides_X,nolandslides['label'],test_size=0.3,random_state=1000)#分训练集和测试集

Xtrain = pd.concat([Xtrain1,Xtrain0],0)
Ytrain = pd.concat([Ytrain1,Ytrain0],0)
Xtest = pd.concat([Xtest1,Xtest0],0)
Ytest = pd.concat([Ytest1,Ytest0],0)

# In[保存test]
#Xtrain1.to_excel('E:/LSM/无监督/landslide_train.xlsx')
#Xtest1.to_excel('E:/LSM/无监督/Xtest1.xlsx')


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
class_report = classification_report(Ytest, Ytest_predict)
print("分类报告：")
print(class_report)

# In[]保存模型
'''
{'max_depth': 42,
 'max_features': 2,
 'min_samples_split': 10,
 'n_estimators': 176}
'''
#保存模型到文件 joblib.dump(rf, 'E:/LSM/无监督/rf.pkl')

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

rf = joblib.load('E:/LSM/无监督/rf.pkl')
#Y_result = rf.predict(X_result)
# 获取测试集样本的连续概率值
Y_result_proba = rf.predict_proba(X_result)

df_data['label'] = Y_result_proba[:,1]

df['label'] = df_data['label']

label = np.array(df['label']).reshape(1,3880, 4160)

df_nc = temp
df_nc.data = label

with rasterio.Env():
    df_nc.rio.to_raster('E:/LSM/无监督/RF_LSI.tif')
df_nc.plot()















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


#混淆矩阵图

def plot_Confusion_Matrix(Ytest, Ytest_predict):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    plt.rcParams['savefig.dpi'] = 300 #图片像素(根据需求调,值太小图看不清)
    plt.rcParams['figure.dpi'] = 300 
    
    # 计算混淆矩阵
    cm = confusion_matrix(Ytest, Ytest_predict)
    
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


plot_Confusion_Matrix(Ytest, Ytest_predict)





