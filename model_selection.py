import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import scipy.stats
import json
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os


result_dir_path = './result'
"""
--------------------------------
加载数据
"""
X_train = pd.read_csv(os.path.join(result_dir_path, 'X_train.csv'))
Y_train = pd.read_csv(os.path.join(result_dir_path, 'Y_train.csv'))
X_valid = pd.read_csv(os.path.join(result_dir_path, 'X_valid.csv'))
Y_valid = pd.read_csv(os.path.join(result_dir_path, 'Y_valid.csv'))

Y_train = np.array(Y_train).reshape((-1, ))
Y_valid = np.array(Y_valid).reshape((-1, ))

"""
--------------------------------
建立baseline
"""
baseline_accuracy = 1 - np.sum(Y_valid)/Y_valid.shape[0]
print('\n')
print('-' * 100)
print('The baseline guess is %d' % 0)
print("Baseline Performance on the test set: accuracy = %0.4f" % baseline_accuracy)


"""
--------------------------------
使用默认参数、交叉验证对比模型
"""
# 各种分类算法
classifiers = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(n_estimators=10),
    ensemble.BaggingClassifier(n_estimators=10),
    ensemble.ExtraTreesClassifier(n_estimators=10),
    ensemble.GradientBoostingClassifier(n_estimators=10),
    ensemble.RandomForestClassifier(n_estimators=10),

    #     Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(cv=3, max_iter=1000),
    linear_model.PassiveAggressiveClassifier(max_iter=1000, tol=1e-3),
    linear_model.RidgeClassifierCV(cv=3),
    linear_model.SGDClassifier(max_iter=1000, tol=1e-3),
    linear_model.Perceptron(max_iter=1000, tol=1e-3),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True, gamma='scale'),
    svm.NuSVC(probability=True, gamma='scale'),
    #     svm.LinearSVC(max_iter=1000),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost
    XGBClassifier()
]

# 统计各个算法对于训练数据的交叉验证的准确率
cv_results = pd.DataFrame(columns=['clasifier name', 'mean accuracy', 'accuracy std'])
row_index = 0
for model in classifiers:
    score = model_selection.cross_val_score(model, X_train, Y_train, scoring="accuracy", cv=10, n_jobs=4)
    cv_results.loc[row_index, 'clasifier name'] = model.__class__.__name__
    cv_results.loc[row_index, 'mean accuracy'] = score.mean()
    cv_results.loc[row_index, 'accuracy std'] = score.std()
    row_index += 1
cv_results.sort_values(by=['mean accuracy'], ascending=False, inplace=True)

print('\n')
print('-' * 100)
print('accuracy rate of cross validation:')
print(cv_results)

# 将交叉验证的准确率绘制成图表

plt.style.use({'figure.figsize': (8, 6)})
sns.set(font_scale=1)
sns.barplot(x='mean accuracy', y='clasifier name', data=cv_results)
plt.title('Cross validation Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Model')
plt.savefig(os.path.join(result_dir_path, "models_cross_validation_comparison.png"), bbox_inches='tight')
# plt.pause(1)

"""
--------------------------------
在参数空间中随机进行参数搜索，对比模型
"""
# 分类算法
classifiers = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(n_estimators=10),
    ensemble.BaggingClassifier(n_estimators=10),
    ensemble.ExtraTreesClassifier(n_estimators=10),
    ensemble.GradientBoostingClassifier(n_estimators=10),
    ensemble.RandomForestClassifier(n_estimators=10),

    # SVM
    svm.SVC(probability=True, gamma='scale'),

    # xgboost
    XGBClassifier()
]

# 设置进行随机搜索的参数空间
learning_rate = scipy.stats.uniform(loc=0.01, scale=1.0 - 0.01)
ratio = scipy.stats.uniform(loc=0.1, scale=1.0 - 0.1)
min_samples_leaf = [1, 2, 4, 6, 8]
min_samples_split = [2, 4, 6, 10]
cv = [3, 5, None]
max_depth = [2, 4, 6, 8, 10, None]
max_features = ['auto', 'sqrt', 'log2', None]
bool_value = [True, False]

hyperparameter_list = [
    # AdaBoostClassifier
    {'n_estimators': scipy.stats.randint(low=10, high=1000), 'algorithm': ['SAMME', 'SAMME.R'],
     'learning_rate': learning_rate},

    # BaggingClassifier
    {'n_estimators': scipy.stats.randint(low=10, high=400), 'max_samples': ratio},

    # ExtraTreesClassifier
    {'n_estimators': scipy.stats.randint(low=10, high=500), 'criterion': ['gini', 'entropy'], 'max_depth': max_depth,
     'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_features': max_features},

    # GradientBoostingClassifier
    {'n_estimators': scipy.stats.randint(low=10, high=700), 'loss': ['deviance', 'exponential'],
     'learning_rate': learning_rate,
     'criterion': ['friedman_mse', 'mse', 'mae'], 'min_samples_split': min_samples_split,
     'min_samples_leaf': min_samples_leaf,
     'max_depth': max_depth, 'max_features': max_features},

    # RandomForestClassifier
    {'n_estimators': scipy.stats.randint(low=10, high=300), 'max_depth': max_depth,
     'min_samples_leaf': min_samples_leaf,
     'min_samples_split': min_samples_split, 'max_features': max_features, },

    # SVC
    {'C': [1, 2, 3, 4, 5], 'gamma': ratio, 'decision_function_shape': ['ovo', 'ovr'], 'probability': [True]},

    # XGBClassifier
    {'n_estimators': scipy.stats.randint(low=10, high=300), 'learning_rate': learning_rate,
     'max_depth': [2, 4, 6, 8, 10]}
]


# 将np.darray转化为list，方便存储在json格式中
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# 创建存储随机搜索结果dataframe
random_cv_columns = ['model name', 'model parameters', 'model train accuracy mean', 'model test accuracy mean',
                     'model test accuracy std']
models_random_cv_comparison = pd.DataFrame(columns=random_cv_columns)
best_model_list = []
row_index = 0

# 对每一个分类算法进行随机参数搜索
for model, param in zip(classifiers, hyperparameter_list):
    random_search = model_selection.RandomizedSearchCV(estimator=model, iid=False,
                                                       param_distributions=param,
                                                       cv=10, n_iter=100, scoring='accuracy',
                                                       n_jobs=6, verbose=1,
                                                       return_train_score=True,
                                                       random_state=0)
    model_name = model.__class__.__name__
    print('random_search for: %s' % model_name)
    random_search.fit(X_train, Y_train)

    # 将找到的每种算法的最好的参数、准确率保存起来
    random_search_results = random_search.cv_results_
    best_index = random_search.best_index_
    models_random_cv_comparison.loc[row_index, 'model name'] = model_name
    models_random_cv_comparison.loc[row_index, 'model parameters'] = json.dumps(random_search.best_estimator_.get_params(),
                                                                                cls=NumpyEncoder)
    models_random_cv_comparison.loc[row_index, 'model train accuracy mean'] = random_search_results['mean_train_score'][
        best_index]
    models_random_cv_comparison.loc[row_index, 'model test accuracy mean'] = random_search_results['mean_test_score'][best_index]
    models_random_cv_comparison.loc[row_index, 'model test accuracy std'] = random_search_results['std_test_score'][best_index]
    best_model_list.append(random_search.best_estimator_)
    row_index += 1

# 按测试准确率对各个分类算法进行排序
models_random_cv_comparison.sort_values(by=['model test accuracy mean'], ascending=False, inplace=True)

print('\n')
print('-' * 100)
print('comparison of models random cv in parameter space')
print(models_random_cv_comparison)
models_random_cv_comparison.to_csv(os.path.join(result_dir_path, 'models_random_cv_comparison.csv'), index=False)

plt.style.use({'figure.figsize': (8, 6)})
sns.set(font_scale=1)
sns.barplot(x='model test accuracy mean', y='model name', data=models_random_cv_comparison)
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
plt.savefig(os.path.join(result_dir_path, "models_random_cv_comparison.png"), bbox_inches='tight')

"""
--------------------------------
绘制模型的学习曲线
"""


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # model_selection.learning_curve
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# 删除已有的学习曲线图像
files_name = os.listdir(result_dir_path)
for file in files_name:
    if file.startswith('learning_curve'):
        img_file_path = os.path.join(result_dir_path, file)
        os.remove(img_file_path)
        print('remove image: %s' % img_file_path)


plt.style.use({'figure.figsize': (8, 6)})
for best_model in best_model_list:
    model_name = best_model.__class__.__name__
    classifier_detail = models_random_cv_comparison[models_random_cv_comparison['model name'].isin([model_name])]
    params = json.loads(classifier_detail.iloc[0, 1])
    if 'n_estimators' in params:
        title = model_name + '[n_estimators(%d)]' % params['n_estimators']
    else:
        title = model_name
    plot_learning_curve(best_model, title, X_train, Y_train, cv=10)
    plt.savefig(os.path.join(result_dir_path, "learning_curve_%s.png" % title), bbox_inches='tight')
