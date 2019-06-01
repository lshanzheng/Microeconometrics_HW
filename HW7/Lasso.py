#_*_ coding:utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV


# config
test_seed = 10000
test_nums = 1000000
train_nums = 1000
train_epochs = 1000
max_iter = 10
model = 1

# set testset

if model == 1:
    n_features = 50
    np.random.seed(test_seed)
    x_test = np.random.randn(test_nums,n_features)
    # set coef
    coef = np.zeros(n_features)
    # alpha1,3...17,19 U(-2,2)
    np.random.seed(test_seed)
    coef[0:20:2] = np.random.uniform(-2,2,size=(10))
    # alpha21,23...n_features-1,n_features N(0,1)
    np.random.seed(test_seed)
    coef[20:n_features:2] = np.random.rand((n_features-20)//2)

if model == 2:
    n_features = 60
    np.random.seed(test_seed)
    x_test = np.random.randn(test_nums,n_features)
    # set coef
    coef = np.zeros(n_features)
    # alpha1,2...19,20 U(-2,2)
    np.random.seed(test_seed)
    coef[0:20:2] = np.random.uniform(-2,2,size=(10))
    # alpha21,23...n_features-1,n_features N(0,1)
    np.random.seed(test_seed)
    coef[20:n_features:2] = np.random.rand((n_features-20)//2)

if model == 3:
    n_features = 50
    np.random.seed(test_seed)
    x_test = np.random.randn(test_nums,n_features)
    # set coef
    coef = np.zeros(n_features)
    # alpha1,2...19,20 U(-2,2)
    np.random.seed(test_seed)
    coef[0:20:2] = np.random.uniform(-2,2,size=(10))
    # alpha21,23...47,49 N(0,1)
    #np.random.seed(test_seed)
    #coef[20:50:2] = np.random.rand(15)

# set y
y_test = np.dot(x_test,coef)

# add noisy
np.random.seed(test_seed)
y_test += 0.001*np.random.rand(test_nums)

# LassoCV: 基于坐标下降法的Lasso交叉验证,这里使用20折交叉验证法选择最佳alpha
# set alpha = 0.002
'''
np.random.seed(test_seed)
X = np.random.randn(10000,n_features)
np.random.seed(test_seed)
Y = np.dot(X,coef)+0.001*np.random.randn(10000)
model = LassoCV(cv=20).fit(X, Y)
alpha = model.alpha_
print('best alpha:',alpha)
'''

# one lasso
mean_error=[]
coefs = []
index = np.arange(1,n_features+1)
fig = plt.figure()
for i in range(train_epochs):
    np.random.seed(i)
    x_train = np.random.randn(train_nums,n_features)
    np.random.seed(i)
    y_train = np.dot(x_train,coef)+0.001*np.random.randn(train_nums)
    lasso = Lasso(max_iter=max_iter, alpha=0.002)
    y_pred = lasso.fit(x_train,y_train).predict(x_test)
    mean_error.append(np.mean((y_test-y_pred)**2))
    coefs.append(lasso.coef_)
    #print('iteration %d mean_error: %.9f'%(i+1,mean_error[i]))
    l1 = plt.scatter(index,lasso.coef_,marker='o',c='b',alpha=0.1)

print('all iterations mean error: %.9f\n' % np.min(mean_error))
print('the best coef:',coefs[mean_error.index(np.min(mean_error))])
print('the raw coef:',coef)
l2 = plt.scatter(index,coef,marker='+',c='r',alpha=1)
plt.title('lasso params plot')
plt.xlabel('coef index')
plt.ylabel('coef value')
plt.xlim(0,n_features+1)
plt.ylim(-2.5,2.5)
plt.legend([l1, l2], ['estimate params', 'raw params'], loc = 'upper right')
plt.savefig('lasso.png')

# post lasso ols
mean_error=[]
coefs = []
index = np.arange(1,n_features+1)
fig = plt.figure()
for i in range(train_epochs):
    np.random.seed(i)
    x_train = np.random.randn(train_nums,n_features)
    np.random.seed(i)
    y_train = np.dot(x_train,coef)+0.001*np.random.randn(train_nums)
    # post-lasso ols
    lasso = Lasso(max_iter=max_iter, alpha=0.002)
    lasso.fit(x_train,y_train)
    x_train = x_train[:,lasso.coef_!=0]
    ols = LinearRegression()
    ols.fit(x_train,y_train)
    y_pred = ols.predict(x_test[:,lasso.coef_!=0])
    coef_i = np.zeros(n_features)
    coef_i[lasso.coef_!=0] = ols.coef_
    mean_error.append(np.mean((y_test-y_pred)**2))
    coefs.append(coef_i)
    #print('iteration %d mean_error: %.9f'%(i+1,mean_error[i]))
    l1 = plt.scatter(index,lasso.coef_,marker='o',c='b',alpha=0.1)

print('all iterations mean error: %.9f\n' % np.min(mean_error))
print('the best coef:',coefs[mean_error.index(np.min(mean_error))])
print('the raw coef:',coef)
l2 = plt.scatter(index,coef,marker='+',c='r',alpha=1)
plt.title('post-lasso ols params plot')
plt.xlabel('coef index')
plt.ylabel('coef value')
plt.xlim(0,n_features+1)
plt.ylim(-2.5,2.5)
plt.legend([l1, l2], ['estimate params', 'raw params'], loc = 'upper right')
plt.savefig('post-lasso-ols.png')
plt.show()
