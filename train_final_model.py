import pandas as pd
import json
import os
from sklearn import ensemble, model_selection
from xgboost import XGBClassifier
import numpy as np


result_dir_path = './result'
"""
--------------------------------
加载数据
"""
models_random_cv_comparison = pd.read_csv(os.path.join(result_dir_path, 'models_random_cv_comparison.csv'))
X_train = pd.read_csv(os.path.join(result_dir_path, 'X_train.csv'))
Y_train = pd.read_csv(os.path.join(result_dir_path, 'Y_train.csv'))
X_valid = pd.read_csv(os.path.join(result_dir_path, 'X_valid.csv'))
Y_valid = pd.read_csv(os.path.join(result_dir_path, 'Y_valid.csv'))

Y_train = np.array(Y_train).reshape((-1, ))
Y_valid = np.array(Y_valid).reshape((-1, ))

processed_test_data = pd.read_csv(os.path.join(result_dir_path, 'processed_test_data.csv'))
test_data_passengerId = pd.read_csv(os.path.join(result_dir_path, 'test_data_passengerId.csv'))

"""
--------------------------------
打印之前cv_random随机搜索到的参数
"""

classifier_name = 'RandomForestClassifier'
classifier_detail = models_random_cv_comparison[models_random_cv_comparison['model name'].isin([classifier_name])]
params = json.loads(classifier_detail.iloc[0, 1])

print('-'*100)
print('best params for %s in random cv:' % classifier_name)
print(params)


"""
--------------------------------
进行grid search
"""

n_estimators = list(range(max(params['n_estimators']-50, 1), params['n_estimators']+50, 1))
hyperparameter_grid = {'n_estimators': n_estimators}

if classifier_name == 'RandomForestClassifier':
    selected_classifier = ensemble.RandomForestClassifier(**params)
elif classifier_name == 'XGBClassifier':
    selected_classifier = XGBClassifier(**params)
elif classifier_name == 'GradientBoostingClassifier':
    selected_classifier = ensemble.GradientBoostingClassifier(**params)
elif classifier_name == 'ExtraTreesClassifier':
    selected_classifier = ensemble.ExtraTreesClassifier(**params)
elif classifier_name == 'AdaBoostClassifier':
    selected_classifier = ensemble.AdaBoostClassifier(**params)
else:
    raise RuntimeError('wrong classifier name')

grid_search = model_selection.GridSearchCV(estimator=selected_classifier,
                                           param_grid=hyperparameter_grid,
                                           cv=10, scoring='accuracy',
                                           verbose=1, n_jobs=6,
                                           return_train_score=True)

grid_search.fit(X_train, Y_train)
best_mean_test_score = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
print('best mean test score of grid search result is %0.4f' % best_mean_test_score)
print('best estimator in grid search:')
print(grid_search.best_estimator_)


"""
--------------------------------
在验证集上评估模型
"""


def cal_accuracy(y_true, y_pred):
    return 1 - np.sum(abs(y_true - y_pred)) / y_pred.shape[0]


final_model = grid_search.best_estimator_
final_pred = final_model.predict(X_valid)
accuracy = cal_accuracy(Y_valid, final_pred)
print('Final model performance on the test set:  accuracy = %0.4f.' % accuracy)


"""
--------------------------------
生成测试集上的分类结果
"""
test_label = final_model.predict(processed_test_data)
test_result = pd.DataFrame(columns=['PassengerId', 'Survived'])
test_result['PassengerId'] = test_data_passengerId['PassengerId']
test_result['Survived'] = test_label
test_result.to_csv('final_model_prediction.csv', index=False)
