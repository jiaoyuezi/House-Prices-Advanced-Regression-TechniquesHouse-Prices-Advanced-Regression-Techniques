import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
#warnings.filterwarnings('ignore')
#%matplotlib inline

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
train.info()

# 1 对 Y 进行分析****************************************
# 查看 Y ：train['SalePrice'] 分布
sns.distplot(train['SalePrice'] , fit=norm)     #displot集合了直方图和拟合曲线
(mu, sigma) = norm.fit(train['SalePrice'])     #求出正太分布的均值和标准差
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#legend目的显示图例
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')     
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# 修正 Y ：train['SalePrice']
train["SalePrice"] = np.log1p(train["SalePrice"])   #log1p(x) := log(1+x)
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

df= train.copy()
#df.drop(['SalePrice'],axis =1 ,inplace=True)
# ****************************************

# 2 EDA分析****************************************
# 看  train 的 correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#train['OverallQual']
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['OverallQual'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQual', fontsize=13)
plt.show()
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=10, ymax=15);

# train['GrLivArea']
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea ', fontsize=13)
plt.show()

train = train.drop(train[(train['GrLivArea']>4000) 
                        & (train['SalePrice']< 12.5 )] .index)
GGGGG = train.shape[0]
# train['GarageCars ']
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['GarageCars'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageCars', fontsize=13)
plt.show()

# train['GarageArea']   ，0 很特殊
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['GarageArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)
plt.show()

train = train.drop(train[(train['GarageArea']>1200) 
                        & (train['SalePrice']< 13 )] .index)

# train['FullBath']  ，0 很特殊
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['FullBath'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('FullBath', fontsize=13)
plt.show()

# train['TotalBsmtSF']  ，0 很特殊
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()

# train['1stFlrSF']
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['1stFlrSF'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('1stFlrSF', fontsize=13)
plt.show()

# train['YearBuilt']  ，1900之前
fig, ax = plt.subplots()     #建立画布，默认一幅图
#ax.scatter(x = train['YearBuilt'], y = train['SalePrice'])
sns.boxplot(train.YearBuilt, train.SalePrice)
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('YearBuilt', fontsize=13)
plt.show()

# train['YearRemodAdd'] ,1950 很特殊
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['YearRemodAdd'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('YearRemodAdd', fontsize=13)
plt.show()

# train['GarageArea']
fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['MSSubClass'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)
plt.show()

#Top & saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#进一步分析
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();

# ****************************************

# 3 缺失值处理****************************************
# 3.1 删去 缺失率 > 0.80 及没有意义的的特征值
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])

all_data["LotAreaCut"] = pd.qcut(all_data.LotArea,10)
all_data.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])

all_data.groupby('Utilities').count()
all_data.drop('Utilities',axis =1 ,inplace = True)
all_data.drop('Id',axis =1 ,inplace = True)

all_data_na = (all_data.isnull().sum() / len(all_data)) 
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

all_data.drop((missing_data[missing_data['Missing Ratio'] > 0.80]).index,1,inplace =True)


# 3.2 对于缺失率在 0.05- 0.80  的特征值
# 利用 'LotArea','Neighborhood' 中位数填补
all_data['LotFrontage'] = all_data.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data['LotFrontage'] = all_data.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

# 3.3 针对缺失值相对较小 （0.01-0.05）的车库Garage、地下室Bsmt这两个大类，类别使用NA填充，数据用0填充 
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MSZoning"] = all_data["MSZoning"].fillna("None") 
   
# 3.4 对余下缺失值不足0.01%的数据特征，直接无脑输出统一使用出现最多的属性填充。使用.mode()[0]
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].mode()[0])
all_data['MasVnrType'] = all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])
all_data['Functional'] = all_data["Functional"].fillna(all_data['Functional'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
# ****************************************

# 4 特征工程
# 4.1 将数值型转化为类别型
NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    all_data[col]=all_data[col].astype(str)
# 4.2 编码，精细分类，按照 mean分组   
# MSSubClass 建筑类 
all_data.groupby(['MSSubClass'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

# MSZoning 城市总体规划分区 
all_data.groupby(['MSZoning'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)
# Neighborhood
all_data.groupby(['Neighborhood'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

# Condition1
all_data.groupby(['Condition1'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

# BldgType
all_data.groupby(['BldgType'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

# HouseStyle
all_data.groupby(['HouseStyle'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

# Exterior1st
all_data.groupby(['Exterior1st'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

# MasVnrType
all_data.groupby(['MasVnrType'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

# Foundation
all_data.groupby(['Foundation'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

# MSZoning
all_data.groupby(['MSZoning'])[['SalePrice']].agg(
        ['mean','median','count']).sort_values(
        by=('SalePrice','mean'),ascending=False)

#编码
def map_values():
    all_data["Exterior1st"] = all_data.Exterior1st.map(
        {      'BrkComm':1,
               'AsphShn':2, 'CBlock':2, 'AsbShng':2,
               'WdShing':3, 'Wd Sdng':3, 'MetalSd':3, 'Stucco':3, 'HdBoard':3,
                'BrkFace':4, 'Plywood':4,
                'VinylSd':5,'CemntBd':5,
                'Stone':7, 'ImStucc':7})
    
    all_data["MSSubClass"] = all_data.MSSubClass.map(
        {       '150':1,
                '30':2, '45':2, '180':2,
                '190':3, 
                 '160':4, '90':4, '50':4, '40':4, '85':4, '70':4,
                 '20':5, '75':5, '80':5, 
                '120': 6, '60':6})
    
    all_data["MSZoning"] = all_data.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4, 'None':5})
    
    all_data["Neighborhood"] = all_data.Neighborhood.map(
        {      'IDOTRR':1,'MeadowV':1,'BrDale':1,
               'OldTown':2, 'Edwards':2,'Sawyer':2 , 'BrkSide':2,  'Blueste':2,'SWISU':2,                               
               'Mitchel':3,'NAmes':3, 'NPkVill':3, 
               'ClearCr':4,'Crawfor':4, 'Blmngtn':4, 'SawyerW':4,'CollgCr':4,'Gilbert':4,'NWAmes':4,
               'Timber':5,'Veenker':5,'Somerst':5, 
               'NridgHt':6,'StoneBr':6,  
                'NoRidge':7})
    
    all_data["Condition1"] = all_data.Condition1.map(
        {     'Artery':1,'Feedr':1,
              'RRAe':2,
              'Norm':3, 'RRAn':3,
               'PosN':4, 'RRNe':4,'RRNn':4,
              'PosA':5 })
    
    all_data["BldgType"] = all_data.BldgType.map({
        '2fmCon':1, 'Duplex':1, 'Twnhs':1, '1Fam':2, 'TwnhsE':2})
    
    all_data["HouseStyle"] = all_data.HouseStyle.map(
        {    '1.5Unf':1, 
             '1.5Fin':2,'SFoyer':2, 
             '2.5Unf':3,'1Story':3, 'SLvl':3,
             '2Story':4, '2.5Fin':4})
    all_data["MasVnrType"] = all_data.MasVnrType.map({'BrkCmn':1, 'None':1, 'BrkFace':2, 'Stone':3})
    
    all_data["ExterQual"] = all_data.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    all_data["Foundation"] = all_data.Foundation.map({'Slab':1, 
                                           'BrkTil':2, 'CBlock':2, 'Stone':2,
                                           'Wood':3, 'PConc':4})
    
    all_data["BsmtQual"] = all_data.BsmtQual.map({'Fa':2, 'None':1, 'TA':3, 'Gd':4, 'Ex':5})
    
    all_data["BsmtExposure"] = all_data.BsmtExposure.map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})
    
    all_data["Heating"] = all_data.Heating.map({'Floor':1, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':5})
    
    all_data["HeatingQC"] = all_data.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    all_data["KitchenQual"] = all_data.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    all_data["Functional"] = all_data.Functional.map({'Maj2':1, 'Maj1':2, 'Min1':2, 'Min2':2, 'Mod':2, 'Sev':2, 'Typ':3})
    
    all_data["FireplaceQu"] = all_data.FireplaceQu.map({'None':2, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    all_data["GarageFinish"] = all_data.GarageFinish.map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})
    
    all_data["PavedDrive"] = all_data.PavedDrive.map({'N':1, 'P':2, 'Y':3})
    
    all_data["SaleType"] = all_data.SaleType.map({'COD':1, 'ConLD':1, 'ConLI':1, 'ConLw':1, 'Oth':1, 'WD':1,
                                       'CWD':2, 'Con':3, 'New':3})
    
    all_data["SaleCondition"] = all_data.SaleCondition.map({'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4})            
     
    all_data["GarageType"] = all_data.GarageType.map({'CarPort':1, 'None':1,
                                           'Detchd':2,
                                           '2Types':3, 'Basment':3,
                                            'Attchd':4, 'BuiltIn':5})
    
                                                       
# drop two unwanted columns
    all_data.drop("LotAreaCut",axis=1,inplace=True)
    all_data.drop(['SalePrice'],axis=1,inplace=True)
    return "Done!"

map_values()
                                                            
all_data.info()
# ****************************************

# 5 pipeline ****************************************
# 5.1 Label Encoding three "Year" features.
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):  #训练算法，设置内部参数。
        return self
    
    def transform(self,X):   #数据转换
        lab=LabelEncoder() # 把字符串类型的数据转化为整型
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X
 
# 5.2 修正数据       
        
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self,skew=0.5):
        self.skew = skew
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_numeric=X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X
# 每一步都用元组（ ‘名称’，步骤）来表示
pipe = Pipeline([
    ('labelenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1)),
    ])

data = all_data.copy()
data_pipe = pipe.fit_transform(data)
data_pipe.shape

# 5.3 使用稳健性标量，适应数据包含许多异常值的情形
scaler = RobustScaler()

n_train=train.shape[0] #矩阵第一维度的长度
X = data_pipe[:n_train]  # train data
test_X = data_pipe[n_train:]  # test data
y= train.SalePrice
X_scaled = scaler.fit(X).transform(X)
test_X_scaled = scaler.fit(test_X).transform(test_X)

# 6 特征选择
# 6.1 随机森林重要性排序
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_scaled, y.astype('int'))

from sklearn import preprocessing

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

RD= pd.DataFrame({"Feature Importance":importances }, index=data_pipe.columns)
RD.sort_values("Feature Importance",ascending=False)

RD[RD["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()

class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self,additional=1):
        self.additional = additional
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.additional==1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
        #TotalBsmtSF 地下室面积总计面积 
        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["MSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["MSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["MSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["Neighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["Neighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["Neighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]
            
            X["-_oFunctional_TotalHouse"] = X["Functional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["Functional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["Condition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["Condition1"] + X["OverallQual"]
            
           
            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"]+X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]
            return X

pipe = Pipeline([
    ('labenc', labelenc()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1)),
    ])
    
full_pipe = pipe.fit_transform(data)
# 6.2 PCA 分析

#train.shape[0]
X = full_pipe[:n_train]
test_X = full_pipe[n_train:]
y= train.SalePrice

X_scaled = scaler.fit(X).transform(X)
test_X_scaled = scaler.transform(test_X)

pca = PCA(n_components = 190)
##tiao zheng
X_scaled=pca.fit_transform(X_scaled)
test_X_scaled = pca.transform(test_X_scaled)

X_scaled.shape, test_X_scaled.shape

# define cross validation strategy
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

models = [LinearRegression(),
          Ridge(),
          Lasso(alpha=0.01,max_iter=10000),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          SVR(),
          LinearSVR(),
          ElasticNet(alpha=0.001,max_iter=10000),
          SGDRegressor(max_iter=1000,tol=1e-3),
          BayesianRidge(),
          KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ]

names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR",
        "Ela","SGD","Bay","Ker"]
for name, model in zip(names, models):
    score = rmse_cv(model, X_scaled, y)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))

class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])

grid(Lasso()).grid_get(X_scaled,y,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[10000]})
grid(Ridge()).grid_get(X_scaled,y,{'alpha':[35,40,45,50,55,60,65,70,80,90]})
grid(SVR()).grid_get(X_scaled,y,{'C':[11,12,13,14,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})
param_grid={'alpha':[0.2,0.3,0.4,0.5], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1,1.2]}
grid(KernelRidge()).grid_get(X_scaled,y,param_grid)
grid(ElasticNet()).grid_get(X_scaled,y,{'alpha':[0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3],'max_iter':[10000]})
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self,mod,weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w

lasso = Lasso(alpha=0.0006,max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma= 0.0004,kernel='rbf',C=15,epsilon=0.009) 
ker = KernelRidge(alpha=0.5 ,kernel='polynomial',degree=3 , coef0=1)
ela = ElasticNet(alpha=0.005, max_iter=10000)
bay = BayesianRidge()

# assign weights based on their gridsearch score
w1 = 0
w2 = 0
w3 = 0
w4 = 1
w5 = 0.2


weight_avg = AverageWeight(mod = [lasso,ridge,svr,ker,ela],weight=[w1,w2,w3,w4,w5])
rmse_cv(weight_avg,X_scaled,y),  rmse_cv(weight_avg,X_scaled,y).mean()

class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean
    
# must do imputer first, otherwise stacking won't work, and i don't know why.
a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y.values.reshape(-1,1)).ravel()

stack_model = stacking(mod=[lasso,ridge,svr,ker,ela],meta_model=ker)

X_train_stack, X_test_stack = stack_model.get_oof(a,b,test_X_scaled)
X_train_stack.shape, a.shape

X_train_add = np.hstack((a,X_train_stack))
X_test_add = np.hstack((test_X_scaled,X_test_stack))
X_train_add.shape, X_test_add.shape

print(rmse_cv(stack_model,X_train_add,b))
print(rmse_cv(stack_model,X_train_add,b).mean())

# This is the final model I use
stack_model = stacking(mod=[lasso,ridge,svr,ker],meta_model=ker)

stack_model.fit(a,b)
pred = np.exp(stack_model.predict(test_X_scaled))

result=pd.DataFrame({'Id':test.Id, 'SalePrice':pred})
result.to_csv("submission.csv",index=False)






















