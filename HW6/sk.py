# _*_ coding:utf-8 _*_
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd

# read dataset
data = pd.read_excel('./RealEstateValuationDataSet.xlsx',index_col=0)
print(data.info())

# preprocess dataset
data['X1 transaction date'] = data['X1 transaction date'] - 2012
data['X3 distance to the nearest MRT station'] = data['X3 distance to the nearest MRT station'] / 1000
Y = data['Y house price of unit area']
X = data.drop(columns=['Y house price of unit area'])
x = X.values
y = Y.values

print('##################################################################') 
# 随机挑选
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=33)

#数据标准化
ss_x = preprocessing.StandardScaler()
train_x = ss_x.fit_transform(train_x)
test_x = ss_x.transform(test_x)
 
ss_y = preprocessing.StandardScaler()
train_y = ss_y.fit_transform(train_y.reshape(-1, 1))
test_y = ss_y.transform(test_y.reshape(-1, 1))

# 多层感知器-回归模型 参数选择 3,26,28
'''
print('###############################参数网格优选###################################')
def genrator():
    for i in range(3,10):
        for j in range(3,30):
            for k in range(3,30):
                yield (i,j,k)
best_score = 0
best_list = []
num = 0
for i,j,k in genrator():
    model_mlp = MLPRegressor(hidden_layer_sizes=(i,j,k),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_mlp.fit(train_x,train_y.ravel())
    mlp_score=model_mlp.score(test_x,test_y.ravel())
    num = num+1
    if num%30 == 0:
        print('num:',num,i,j,k)
    if mlp_score>70.0:
        print('sklearn MLP',mlp_score)
        print(i,j,k)
    if mlp_score > best_score:
        best_list = [i,j,k]
        print('new_score',mlp_score,'and list',best_list)
        best_score = mlp_score
print('best_list',best_list)
'''

# 集成-回归模型 参数选择
'''
model_gbr=GradientBoostingRegressor()
model_gbr.fit(train_x,train_y.ravel())
gbr_score_disorder=model_gbr.score(test_x,test_y.ravel())
print('sklearn ensemble',gbr_score_disorder)
print('###############################参数网格优选###################################')
model_gbr_GridSearch=GradientBoostingRegressor()
#设置参数池  参考 http://www.cnblogs.com/DjangoBlog/p/6201663.html
param_grid = {'n_estimators':range(20,81,10),
              'learning_rate': [0.2,0.1, 0.05, 0.02, 0.01 ],
              'max_depth': [4, 6,8],
              'min_samples_leaf': [3, 5, 9, 14],
              'max_features': [0.8,0.5,0.3, 0.1]}
#网格调参
from sklearn.model_selection import GridSearchCV
estimator = GridSearchCV(model_gbr_GridSearch,param_grid )
estimator.fit(train_x,train_y.ravel() )
print('最优调参：',estimator.best_params_)
# {'learning_rate': 0.1, 'max_depth': 4, 'max_features': 0.5, 'min_samples_leaf': 14, 'n_estimators': 30}
print('调参后得分',estimator.score(test_x, test_y.ravel()))
'''

model_mlp_best = MLPRegressor(hidden_layer_sizes=(3,26,28),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model_mlp_best.fit(train_x,train_y.ravel())
mlp_score=model_mlp_best.score(test_x,test_y.ravel())
print('sklearn MLP',mlp_score) #准确率 0.713

model_gbr_best=GradientBoostingRegressor(learning_rate=0.1,max_depth=4,max_features=0.5,min_samples_leaf=14,n_estimators=30)
model_gbr_best.fit(train_x,train_y.ravel() )
gbr_score=model_gbr_best.score(test_x,test_y.ravel())
print('sklearn ensemble',gbr_score)#准确率 0.702

#使用最好的集成模型进行预测
gbr_pridict=model_gbr_best.predict(test_x)
#多层感知器
mlp_pridict=model_mlp_best.predict(test_x)
 
#画图
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(1, 1, 1)
line3,=axes.plot(range(len(test_y)), test_y, 'g',label='GroundTruth')
line1,=axes.plot(range(len(gbr_pridict)), gbr_pridict, 'b--',label='Ensemble',linewidth=2)
line2,=axes.plot(range(len(mlp_pridict)), mlp_pridict, 'r--',label='MLP',linewidth=2)
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1, line2, line3])
plt.title("sklearn Regression")
plt.show()
#print(ss_y.inverse_transform(mlp_pridict))
