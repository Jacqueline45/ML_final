import os
import csv
#import glob
import numpy as np
import pandas as pd
#from google.colab import files
#uploaded = files.upload()
#uploaded1 = files.upload()
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import sys

# load data
data_train = pd.read_csv(sys.argv[1])
data_test = pd.read_csv(sys.argv[2])
data_train1 = data_train.drop(['SalePrice'],axis=1)
print("data_train", data_train.shape)
print("data_train1", data_train1.shape)
print("data_test", data_test.shape)
# concate train/test data
data_all = pd.concat((data_train1, data_test), sort=False).reset_index(drop=True)
print("data_all", data_all.shape)

# numeric feature describe
num_data=data_all.select_dtypes(['int64','float64'])
describe_num=data_all.describe().transpose()
#print (describe_num)

# correlation heatmap
num_train=data_train.select_dtypes(['int64','float64'])
num_corr=num_train.corr().drop('Id')   
# top 10 
#print (num_corr['SalePrice'].sort_values(ascending=False).iloc[1:11])

#categorical feature describe
cat_data=data_all.select_dtypes(['object'])

#遺漏值檢查
missing_columns=data_all.isnull().mean().sort_values(ascending=False)
missing_columns=missing_columns[missing_columns!=0].to_frame().reset_index()

# 移除遺漏值太多或分布太奇怪的欄位
data_all=data_all.drop(columns=['Heating', 'RoofMatl', 'Condition2', 'Utilities', 'Street', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'],
axis=1)
data_train=data_train.drop(columns=['Heating', 'RoofMatl', 'Condition2', 'Utilities', 'Street', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'],
axis=1)
missing_cat=['FireplaceQu','GarageCond','GarageType','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrType','Electrical','MSZoning','Functional','Exterior1st','KitchenQual','Exterior2nd','SaleType']
missing_num=['LotFrontage','MasVnrArea','BsmtFullBath','BsmtHalfBath','TotalBsmtSF','GarageArea','BsmtUnfSF','BsmtFinSF2','GarageCars','BsmtFinSF1']
# 類別補none
for i in missing_cat:
    data_all[i]=data_all[i].fillna('none')
    data_train[i]=data_train[i].fillna('none')
# 數值補0    
for i in missing_num:
    data_all[i]=data_all[i].fillna(0)
    data_train[i]=data_train[i].fillna(0)
# 建造年補中位數
data_all['GarageYrBlt']=data_all['GarageYrBlt'].fillna(data_all['GarageYrBlt'].median())
data_train['GarageYrBlt']=data_train['GarageYrBlt'].fillna(data_train['GarageYrBlt'].median())

# 檢查連續變項跟依變數的分布情形
data_remove_outlier=data_train

outlier_columns=['LotFrontage','MasVnrArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF', 'GrLivArea','BedroomAbvGr','TotRmsAbvGrd', 'MiscVal']
outlier_threshold=[300, 1400, 5000, 6000, 4000, 4500, 8, 14, 8000]
for c,n in zip(outlier_columns,outlier_threshold):
    data_remove_outlier=data_remove_outlier[data_remove_outlier[c]<n]
    data_remove_outlier1 = data_remove_outlier.drop(['SalePrice'],axis=1)
data_all = pd.concat((data_remove_outlier1, data_all[1460:]), sort=False).reset_index(drop=True)

for i in list(['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
               '1stFlrSF','2ndFlrSF','LowQualFinSF','GarageArea','WoodDeckSF',
               'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','GrLivArea',
               'LotArea','PoolArea','MiscVal']):
    data_all[i]=(np.log1p(data_all[i].dropna()))

for i in ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','FireplaceQu',]:
    data_all[i]=data_all[i].replace(['Ex','Gd','TA','Fa','Po', 'none'], [5, 4, 3, 2, 1, 0]).astype(int)
#['Ex','Gd','TA','Fa','Po', 'none'], [5, 4, 3, 2, 1, 0]

for i in ['KitchenQual']:
    data_all[i]=data_all[i].replace(['Ex','Gd','TA','Fa','Po', 'none'], [7, 4, 3, 2, 1, 0]).astype(int)

#data_all[i]=data_all[i].replace(['Ex','Gd','TA','Fa','Po', 'none'], [5, 4, 3, 2, 1, 0]).astype(int)

for i in['GarageQual']:
    data_all[i]=data_all[i].replace(['Ex','Gd','TA','Fa','Po', 'none'], [7, 4, 3, 2, 1, 0]).astype(int)
#data_all[i]=data_all[i].replace(['Ex','Gd','TA','Fa','Po', 'none'], [5, 4, 3, 2, 1, 0]).astype(int)    

for i in ['GarageCond']:
    data_all[i]=data_all[i].replace(['Ex','Gd','TA','Fa','Po', 'none'], [5, 4, 3, 2, 1, 0]).astype(int)
#data_all[i]=data_all[i].replace(['Ex','Gd','TA','Fa','Po', 'none'], [5, 4, 3, 2, 1, 0]).astype(int)

for i in ['BsmtExposure']:
    data_all[i]=data_all[i].replace(['Gd','Av','Mn','No', 'none'], [6, 3, 2, 1, 0]).astype(int)
#['Gd','Av','Mn','No', 'none'], [6, 3, 2, 1, 0] 0.13133
for i in ['BsmtFinType1','BsmtFinType2']:
    data_all[i]=data_all[i].replace(['GLQ','ALQ','BLQ','Rec', 'LwQ','Unf','none'], [6, 5, 4, 3, 2, 1, 0]).astype(int)
#['GLQ','ALQ','BLQ','Rec', 'LwQ','Unf','none'], [6, 5, 4, 3, 2, 1, 0]
# 銷售時屋齡
data_all['House_year']=data_all['YrSold']-data_all['YearBuilt']
# 銷售時屋齡(整修)
data_all['Remod_year']=data_all['YrSold']-data_all['YearRemodAdd']
# 幾年前建造車庫
data_all['Garage_built']=data_all['YrSold']-data_all['GarageYrBlt']

for i in ['YrSold','MSSubClass','MoSold']:
    data_all[i]=data_all[i].astype(str)

# one hot encoding
data_final=pd.get_dummies(data_all)
data_final=data_final.drop(['Id'],axis=1).reset_index(drop=True)
print("data_final", data_final.shape)
# split train and validate data
x = data_final[:1453]
y=np.array(np.log1p(data_remove_outlier['SalePrice']))
Test = data_final[1453:]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=321) #random_state=321
#print(type(x_train))

#print(y_train[0:10])

from sklearn.linear_model import RidgeCV
# set cross-validation alpha
alpha=[0.0001,0.001,0.01,0.1,1,10,100]
# find the best alpha and build model
Ridge = RidgeCV(cv=5, alphas=alpha,normalize=True)
Ridge_fit=Ridge.fit(x_train,y_train)
y_ridge_train=Ridge_fit.predict(x_train)
y_ridge_test=Ridge_fit.predict(x_test)
Test_pred = Ridge_fit.predict(Test)
Test_pred=np.expm1(Test_pred)
# validation( train data and validate data)
print('RMSE_train_Ridge = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_ridge_train))))
print('RMSE_test_Ridge = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_ridge_test))))

with open(sys.argv[3], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['Id', 'SalePrice']
    #print(header)
    csv_writer.writerow(header)
    for i in range(1459):
        row = [1461+i, int(round(Test_pred[i]))]
        csv_writer.writerow(row)
        #print(row)