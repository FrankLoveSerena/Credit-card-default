#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
# 信用卡违约率分析
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('UCI_Credit_Card.csv')
# 数据探索
pd.set_option('display.max_columns', None)
print(data.shape)
print(data.describe())
# 查看下一个月违约率的情况
next_month = data['default.payment.next.month'].value_counts()
print(next_month)
df = pd.DataFrame({'default.payment.next.month': next_month.index, 'values': next_month.values})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize = (6, 6))
plt.title('信用卡违约率客户\n (违约：1，守约：0)')
sns.set_color_codes(palette = 'pastel')
sns.barplot(x = 'default.payment.next.month', y = 'values', data = df)
locs, labels = plt.xticks()
plt.show()
# 特征选择，去掉ID字段和最后一个结果字段
data.drop('ID', axis = 1, inplace = True)
target = data['default.payment.next.month'].values
columns = data.columns.to_list()
columns.remove('default.payment.next.month')
features = data[columns].values
# 30%作为测试集，其他作为训练集
train_x, test_x, train_y, test_y = train_test_split(features, target, test_size = 0.3, stratify = target,
                                                    random_state = 1)
# 构造各种分类器
classifiers = [SVC(random_state = 1),
               DecisionTreeClassifier(random_state = 1),
               RandomForestClassifier(random_state = 1),
               KNeighborsClassifier()]
# 分类器名称
classifier_names = ['svc',
                    'decisiontreeclassifier',
                    'randomforestclassifier',
                    'kneighborsclassifier']
# 分类器参数
classifier_param_grid = [{'svc__C': [1], 'svc__gamma': [0.01]},
                         {'decisiontreeclassifier__max_depth': [6, 9, 11]},
                         {'randomforestclassifier__n_estimators': [3, 5, 6]},
                         {'kneighborsclassifier__n_neighbors': [4, 6, 8]}]


# 对分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score = 'accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = score, cv = 3)
    # 寻找最优的参数和最优的准确率分数
    search = gridsearch.fit(train_x, train_y)
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数：%0.4lf" % search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("准确率%0.4lf" % accuracy_score(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y, predict_y)
    return response


# 调用函数
for model_name, model, model_param_grid in zip(classifier_names, classifiers, classifier_param_grid):
    pipeline = Pipeline([('scaler', StandardScaler()),
                        (model_name, model)])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid)
    print(result)
