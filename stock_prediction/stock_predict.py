import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split#数据分割包

def import_dataset(path):
    return pd.read_csv('msft_stockprices_dataset.csv')

def data_split(data, x_indexes, y_indexes, rate):
    #获取列名
    column_names = data.columns.values
    #['Date', 'High Price', 'Low Price', 'Open Price', 'Close Price ', 'Volume']
    X = data.iloc[:, x_indexes] # 特征为最高价(High Price)、最低价(Low Price)、开盘价(Open Price)以及成交量(Volume)
    y = data.iloc[:, y_indexes] # 结果为收盘价(Close Price)
    #划分训练集和测试集，划分比例为(1-rate):(rate)
    seed = np.random.randint(0, 1000000)
    return train_test_split(X, y, test_size=rate, random_state=seed)

def lr_train(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def validate(model, X_test, y_test, error_rate):
    y_predict = model.predict(X_test)
    d_value = np.abs((y_predict-y_test)/y_test)
    d_value_list = np.array(d_value).tolist()
    count = 0
    for v in d_value_list:
        if v[0] < error_rate:
            count = count + 1
    print('when error rate is {0}, accuracy rate is {1}'.format(error_rate, count/len(y_test)))

if __name__ == '__main__':
    #导入数据集
    path = 'msft_stockprices_dataset.csv'
    data = import_dataset(path)
    #划分训练集合测试集
    X_train, X_test, y_train, y_test = data_split(data, [1,2,3,5], [4], 0.3)
    #训练模型
    model = lr_train(X_train, y_train)
    #验证
    validate(model, X_test, y_test, 0.1)
    validate(model, X_test, y_test, 0.05)
    validate(model, X_test, y_test, 0.01)