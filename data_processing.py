import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os


"""
--------------------------------
设定
"""
plt.rcParams['font.size'] = 24
sns.set(font_scale=1.5)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)
result_dir_path = './result'
"""
--------------------------------
加载数据
"""
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
# train_data.head()

"""
--------------------------------
离群点检测
"""


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # 遍历所有feature
    for col in features:
        # 计算第一四分位数Q1
        Q1 = np.percentile(df[col], 25)
        # 计算第三四分位数Q3
        Q3 = np.percentile(df[col], 75)
        # 计算四分位距IQR
        IQR = Q3 - Q1

        # 设置outlier step
        outlier_step = 1.5 * IQR

        # 找出离群点的位置
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    # 将包含2个以上离群点的乘客信息进行统计
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


# 从 Age, SibSp , Parch and Fare这些字段中检查离群点
outliers_to_drop = detect_outliers(train_data, 2, ["Age", "SibSp", "Parch", "Fare"])
print('\n')
print('-' * 100)
print('outliers are:')
print(outliers_to_drop)

# 将这些含有2个以上的离群点的乘客信息从训练数据中删除
train_data = train_data.drop(outliers_to_drop, axis=0).reset_index(drop=True)


"""
--------------------------------
处理缺失值
"""
# 先打印，查看当前有哪些字段含有缺失值
print('\n')
print('-' * 100)
print('Train columns with null values:\n', train_data.isnull().sum() / train_data.shape[0])
print("-"*10)
print('Test/Validation columns with null values:\n', test_data.isnull().sum() / test_data.shape[0])
print("-"*10)


# data filed has been changed
for dataset in [train_data, test_data]:
    # 使用Age的中值填充缺失值
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    # 使用Embarked中出现次数最多的label填充缺失值
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    # 使用Fare的中值填充缺失值
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

drop_column = ['Cabin', 'PassengerId', 'Ticket']
train_data.drop(drop_column, axis=1, inplace=True)

test_data_passengerId = test_data[['PassengerId']]
test_data.drop(drop_column, axis=1, inplace=True)

# 填充缺失值之后，再统计一遍
print('\n')
print('-' * 100)
print('after filling missing value')
print(train_data.isnull().sum())
print("-" * 10)
print(test_data.isnull().sum())


"""
--------------------------------
对数据进行分析
"""
print('\n')
print('-' * 100)
print('correlations_matrix:')
correlations_matrix = train_data.corr()['Survived'].sort_values()
print(correlations_matrix, '\n')

# 可以查看各个字段（非浮点数类型的字段）的取值与Survived字段的相关性。
print('\n')
print('-' * 100)
print('Correlation of fields with the Survived:')
for x in train_data:
    if train_data[x].dtype != 'float64' and x not in ['Survived', 'Name']:
        print('Survival Correlation by:', x)
        print(train_data[[x, 'Survived']].groupby(x, as_index=False).mean())
        print('-' * 10, '\n')


"""
--------------------------------
特征工程
"""


def create_features_for_data(data):
    # 创造新特征FamilySize
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # 如果FamilySize==0，则IsAlone=True
    data['IsAlone'] = 1
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0

    # 从Name字段中提取每个人的title
    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    # 某些title的人数较少，同一设置为Misc
    title_names = (data['Title'].value_counts() < 10)
    data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    # 安装Fare、Age字段的取值范围，将取值范围划分成多个分段，这样就可以将连续的变量转换成离散的变量了
    data['FareBin'] = pd.qcut(data['Fare'], 4)
    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

    # 将离散的变量的label转化为数字label
    label = LabelEncoder()
    features = ['Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin']
    for f in features:
        data[f + '_Code'] = label.fit_transform(data[f])

    data = data.drop(columns=['Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin', 'Name'])
    return data


# 为train_data和test_data创造新的featrue
train_data = create_features_for_data(train_data)
test_data = create_features_for_data(test_data)

print('\n')
print('-' * 100)
print('After creating new features')
print('train data shape:', train_data.shape)
train_data.head(5)

"""
--------------------------------
特征选择 去除共线特征
"""


def print_collinear_features(x, threshold):
    """
    Objective:
       删除数据帧中相关系数大于阈值的共线特征。 删除共线特征可以帮助模型泛化并提高模型的可解释性。
    Inputs:
        阈值：删除任何相关性大于此值的特征
    Output:
        仅包含非高共线特征的数据帧
    :param x: 
    :param threshold: 
    :return: 
    """
    # 不要删除能源之星得分之间的相关性
    y = x['Survived']
    x = x.drop(columns=['Survived'])

    # 计算相关性矩阵
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # 迭代相关性矩阵并比较相关性
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # 如果相关性超过阈值
            if val >= threshold:
                # 打印有相关性的特征和相关值
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])


print_collinear_features(train_data, 0.6)

# data filed has been changed
# 从数据中删除AgeBin_Code、SibSp、Parch字段
drop_columns = ['AgeBin_Code', 'SibSp', 'Parch']
train_data = train_data.drop(columns=drop_columns)
test_data = test_data.drop(columns=drop_columns)

print('\n')
print('-' * 100)
print('After deleting collinear features')
print('train data shape:', train_data.shape)
train_data.head(5)

"""
--------------------------------
特征缩放
"""
# 对数据中的Fare、Age字段进行缩放
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
train_data[['Fare']] = train_data[['Fare']].apply(max_min_scaler)
train_data[['Age']] = train_data[['Age']].apply(max_min_scaler)

test_data[['Fare']] = test_data[['Fare']].apply(max_min_scaler)
train_data[['Age']] = train_data[['Age']].apply(max_min_scaler)

print('\n')
print('-' * 100)
print('After features scaling')
train_data.head(5)


"""
--------------------------------
划分训练集、验证集
"""

targets = train_data[['Survived']]
features = train_data.drop(columns=['Survived'])

X_train, X_valid, Y_train, Y_valid = train_test_split(features, targets, test_size=0.3, random_state=0)

print('shape of X_train:', X_train.shape)
print('shape of Y_train:', Y_train.shape)
print('shape of X_valid:', X_valid.shape)
print('shape of Y_valid:', Y_valid.shape)

X_train.to_csv(os.path.join(result_dir_path, 'X_train.csv'), index=False)
Y_train.to_csv(os.path.join(result_dir_path, 'Y_train.csv'), index=False)

X_valid.to_csv(os.path.join(result_dir_path, 'X_valid.csv'), index=False)
Y_valid.to_csv(os.path.join(result_dir_path, 'Y_valid.csv'), index=False)

test_data.to_csv(os.path.join(result_dir_path, 'processed_test_data.csv'), index=False)
test_data_passengerId.to_csv(os.path.join(result_dir_path, 'test_data_passengerId.csv'), index=False)

print('\n')
print('-' * 100)
print('save file: X_train.csv, Y_train.csv, X_valid.csv, Y_valid.csv, processed_test_data.csv, '
      'test_data_passengerId.csv')