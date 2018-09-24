# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 14:36:09 2018

@author: hongy
"""
print("My name is Hongyu Zhang")
print("My NetID is: hongyuz2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 

#read the file
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/' 
                 'python-machine-learning-book-2nd-edition' 
                 '/master/code/ch10/housing.data.txt', 
                 header=None, 
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())

#import the package and create the scatterplot matrix 
cols = df.columns
sns.pairplot(df[cols], size=2.5) 
plt.tight_layout() 
plt.show()


#use Seaborn's heatmap function to plot the correlation matrix array as a heat map
#import numpy as np 
cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale=1.5) 
hm = sns.heatmap(cm,             
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 7}, 
                 yticklabels=cols, 
                 xticklabels=cols) 
plt.tight_layout()
plt.show()


# [linear regression model]
class LinearRegressionGD(object):    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta        
        self.n_iter = n_iter    
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []        
        for i in range(self.n_iter):            
            output = self.net_input(X)
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self   
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]    
    def predict(self, X):        
        return self.net_input(X)    
    
X = df[['RM']].values 
y = df['MEDV'].values 

#from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten() 
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)
    
sns.reset_orig() # resets matplotlib style 
plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()   # the GD algorithm converged after the fifth epoch


#define a simple helper function that will plot a scatterplot of 
#the training samples and add the regression line
def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70) 
    plt.plot(X, model.predict(X), color='black', lw=2) 
    return None   


#plot the number of rooms against house price
lin_regplot(X_std, y_std, lr) 
plt.xlabel('Average number of rooms [RM] (standardized)') 
plt.ylabel('Price in $1000s [MEDV] (standardized)') 
plt.show()




# 4 different models
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

#{Linear regression model}
print('\n' * 3)
print('1. Linear regression model')
slr = LinearRegression() 
slr.fit(X, y) 
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_) 

lin_regplot(X, y, slr) 
plt.xlabel('Average number of rooms [RM] (standardized)') 
plt.ylabel('Price in $1000s [MEDV] (standardized)') 
plt.show()

y_train_pred = slr.predict(X_train) 
y_test_pred = slr.predict(X_test) 

plt.scatter(y_train_pred,  y_train_pred - y_train, 
            c='steelblue', marker='o', edgecolor='white',
            label='Training data') 
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white', 
            label='Test data') 
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2) 
plt.xlim([-10, 50]) 
plt.show()
print('MSE train: %.3f, test: %.3f' % 
      (mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % 
      (r2_score(y_train, y_train_pred), 
       r2_score(y_test, y_test_pred)))



#[Ridge regression model]
print('\n' * 3)
print('2. Ridge regression model')
from sklearn.linear_model import Ridge
sr = Ridge(alpha=1.0,normalize=True)
sr.fit(X, y) 
print('Slope: %.3f' % sr.coef_[0])
print('Intercept: %.3f' % sr.intercept_) 

lin_regplot(X, y, sr) 
plt.xlabel('Average number of rooms [RM] (standardized)') 
plt.ylabel('Price in $1000s [MEDV] (standardized)') 
plt.show() 

y_train_pred2 = sr.predict(X_train) 
y_test_pred2 = sr.predict(X_test) 

plt.scatter(y_train_pred2,  y_train_pred2 - y_train, 
            c='steelblue', marker='o', edgecolor='white',
            label='Training data') 
plt.scatter(y_test_pred2,  y_test_pred2 - y_test,
            c='limegreen', marker='s', edgecolor='white', 
            label='Test data') 
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2) 
plt.xlim([-10, 50]) 
plt.show()

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
MSE2_test_scores = []
MSE2_train_scores = []
R22_test_scores = []
R22_train_scores = []
for alpha in alpha_space:
    sr.alpha=alpha
    sr.fit(X,y)
    y_train_pred2 = sr.predict(X_train) 
    y_test_pred2 = sr.predict(X_test)   
    MSE2_test_scores.append(mean_squared_error(y_test,y_test_pred2))
    MSE2_train_scores.append(mean_squared_error(y_train,y_train_pred2))
    R22_test_scores.append(r2_score(y_test, y_test_pred2))
    R22_train_scores.append(r2_score(y_train, y_train_pred2))
plt.plot(alpha_space,MSE2_test_scores)
plt.xlabel('alpha_space')
plt.ylabel('MSE2_test_scores')
plt.show()

plt.plot(alpha_space,MSE2_train_scores)
plt.xlabel('alpha_space')
plt.ylabel('MSE2_train_scores')
plt.show()

plt.plot(alpha_space,R22_test_scores)
plt.xlabel('alpha_space')
plt.ylabel('R22_test_scores')
plt.show()

plt.plot(alpha_space,R22_train_scores)
plt.xlabel('alpha_space')
plt.ylabel('R22_train_scores')
plt.show()


#[Lasso regression model] 
print('\n' * 3)
print('3. Lasso regression model')
from sklearn.linear_model import ElasticNet    
sl = ElasticNet(alpha=1.0, l1_ratio=1)#set the l1_ratio to 1.0, 
#the ElasticNet regressor would be equal to LASSO regression.
sl.fit(X, y) 
print('Slope: %.3f' % sl.coef_[0])
print('Intercept: %.3f' % sl.intercept_) 

lin_regplot(X, y, sl) 
plt.xlabel('Average number of rooms [RM] (standardized)') 
plt.ylabel('Price in $1000s [MEDV] (standardized)') 
plt.show() 

y_train_pred3 = sl.predict(X_train)
y_test_pred3 = sl.predict(X_test)

plt.scatter(y_train_pred3,  y_train_pred3 - y_train, 
            c='steelblue', marker='o', edgecolor='white',
            label='Training data') 
plt.scatter(y_test_pred3,  y_test_pred3 - y_test,
            c='limegreen', marker='s', edgecolor='white', 
            label='Test data') 
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2) 
plt.xlim([-10, 50]) 
plt.show()

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
MSE3_test_scores = []
MSE3_train_scores = []
R23_test_scores = []
R23_train_scores = []
for alpha in alpha_space:
    sl.alpha=alpha
    sl.fit(X,y)
    y_train_pred3 = sl.predict(X_train)
    y_test_pred3 = sl.predict(X_test)
    MSE3_test_scores.append(mean_squared_error(y_test,y_test_pred3))
    MSE3_train_scores.append(mean_squared_error(y_train,y_train_pred3))
    R23_test_scores.append(r2_score(y_test, y_test_pred3))
    R23_train_scores.append(r2_score(y_train, y_train_pred3))
plt.plot(alpha_space,MSE3_test_scores)
plt.xlabel('alpha_space')
plt.ylabel('MSE3_test_scores')
plt.show()

plt.plot(alpha_space,MSE3_train_scores)
plt.xlabel('alpha_space')
plt.ylabel('MSE3_train_scores')
plt.show()

plt.plot(alpha_space,R23_test_scores)
plt.xlabel('alpha_space')
plt.ylabel('R23_test_scores')
plt.show()

plt.plot(alpha_space,R23_train_scores)
plt.xlabel('alpha_space')
plt.ylabel('R23_train_scores')   
plt.show()


#[ElasticNet regression model]
print('\n' * 3)
print('4. ElasticNet regression model')
se = ElasticNet(alpha=1.0, l1_ratio=0.5)
se.fit(X, y) 
print('Slope: %.3f' % se.coef_[0])
print('Intercept: %.3f' % se.intercept_) 

lin_regplot(X, y, se) 
plt.xlabel('Average number of rooms [RM] (standardized)') 
plt.ylabel('Price in $1000s [MEDV] (standardized)') 
plt.show() 

y_train_pred4 = se.predict(X_train) 
y_test_pred4 = se.predict(X_test) 

plt.scatter(y_train_pred4,  y_train_pred4 - y_train, 
            c='steelblue', marker='o', edgecolor='white',
            label='Training data') 
plt.scatter(y_test_pred4,  y_test_pred4 - y_test,
            c='limegreen', marker='s', edgecolor='white', 
            label='Test data') 
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2) 
plt.xlim([-10, 50]) 
plt.show()

# Setup the array of alphas and lists to store scores
l1_space = np.logspace(-4, 0, 50)
MSE4_test_scores = []
MSE4_train_scores = []
R24_test_scores = []
R24_train_scores = []
for l1_ratio in l1_space:
    y_train_pred4 = se.predict(X_train) 
    y_test_pred4 = se.predict(X_test) 
    se.l1_ratio=l1_ratio
    se.fit(X,y)
    MSE4_test_scores.append(mean_squared_error(y_test,y_test_pred4))
    MSE4_train_scores.append(mean_squared_error(y_train,y_train_pred4))
    R24_test_scores.append(r2_score(y_test, y_test_pred4))
    R24_train_scores.append(r2_score(y_train, y_train_pred4))
plt.plot(alpha_space,MSE4_test_scores)
plt.xlabel('l1_ratio')
plt.ylabel('MSE4_test_scores')
plt.show()

plt.plot(alpha_space,MSE4_train_scores)
plt.xlabel('l1_ratio')
plt.ylabel('MSE4_train_scores')
plt.show()

plt.plot(alpha_space,R24_test_scores)
plt.xlabel('l1_ratio')
plt.ylabel('R24_test_scores')
plt.show()

plt.plot(alpha_space,R24_train_scores)
plt.xlabel('l1_ratio')
plt.ylabel('R24_train_scores')   
plt.show()




    
    
    