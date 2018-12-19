import pandas as pd
import os
import re
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import dask.dataframe as dd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve,recall_score,classification_report,mean_squared_error,confusion_matrix
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from contextlib import contextmanager
import time
import random
import datetime
from sklearn.model_selection import train_test_split as model_tts

# preset the data types
set_data_type = {'ip': np.int64,
                 'app': np.int16,
                 'device': np.int16,
                 'os': np.int16,
                 'channel': np.int16,
                 'is_attributed' : np.int16
                }



#visualisation functions
def plot_proportion(column, list_order1, list_order0):
    dict_ratio_os = {}
    for i in list_order1:
        if i in list_order0:
            dict_ratio_os[i] = list_order1[i] / list_order0[i]
    sorted_data = sorted(dict_ratio_os.items(), key = lambda x:x[1], reverse = True)[:10]
    list_order_prop = {str(i[0]):i[1] for i in sorted_data}
    plot = sns.lineplot(x = list(list_order_prop.keys()),y= list(list_order_prop.values()))
    plot.set_title("Proportion of true positives %s"%column)
    plt.show()


def func_get_count (train_sample_column):
    counter_os = Counter(train_sample_column)
    sorted_data = sorted(counter_os.items(), key = lambda x:x[1], reverse = True)
    list_keys_order = [i[0] for i in sorted_data[:10]]
    list_order = {i[0]:i[1] for i in sorted_data}
    return (list_keys_order, list_order)

def draw_plot(column, train_sample_is_attributed_true, train_sample_is_attributed_false):
    list_keys_order, list_order1 = func_get_count(train_sample_is_attributed_true[column])
    plot = sns.countplot(train_sample_is_attributed_true[column], order = list_keys_order)
    plot.set_title("Count of %s with is_attributed = 1" %column)
    plt.show()

    list_keys_order, list_order0 = func_get_count(train_sample_is_attributed_false[column])
    plot = sns.countplot(train_sample_is_attributed_false[column], order = list_keys_order)
    plot.set_title("Count of %s with is_attributed = 0" %column)
    plt.show()
    
    plot_proportion(column, list_order1, list_order0)

def unique_feature(data):
    plt.figure(figsize=(15,10))
    columns = ['ip', 'app', 'device', 'os', 'channel']
    unique_col = [len(data[col].unique()) for col in columns]
    sns.set(font_scale=1.5)
    ax = sns.barplot(columns, unique_col, log=True)
    ax.set(xlabel='Feature', ylabel='Total counts', title='Number of count for per unique features per 100,000')
    for p, uniq in zip(ax.patches, unique_col):
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 10,
                uniq,
                ha="center")

def app_count(data):
    plt.figure(figsize=(6,6))
    #sns.set(font_scale=1.2)
    mean = (data.is_attributed.values == 1).mean()
    ax = sns.barplot(['App Downloaded', 'Not Downloaded'], [mean, 1-mean])
    ax.set(ylabel='Proportion of app downloads', title='App Downloaded vs Not Downloaded')
    for p, uniq in zip(ax.patches, [mean, 1-mean]):
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height+0.01,
                '{}%'.format(round(uniq * 100, 2)),
                ha="center")

def ip_analysis(data):
    #grouping by ip and getting the mean to find proportion
    proportion = data[['ip', 'is_attributed']].groupby('ip', as_index=False).mean().sort_values('is_attributed', ascending=False)
    #counting unique ip for each group
    counts = data[['ip', 'is_attributed']].groupby('ip', as_index=False).count().sort_values('is_attributed', ascending=False)
    merge_table = counts.merge(proportion, on='ip', how='left')
    merge_table.columns = ['ip', 'click_count_per_group', 'prop_download']

    ax = merge_table[:300].plot(secondary_y='prop_download')
    plt.title('Download rates of most frequent IPs')
    ax.set(ylabel='Clicks count')
    plt.ylabel('Proportion Downloaded')
    plt.show()

    print('Download Rates over Counts of Most Popular IPs')
    print(merge_table[:20])

def app_analysis(data):
    proportion = data[['app', 'is_attributed']].groupby('app', as_index=False).mean().sort_values('is_attributed', ascending=False)
    counts = data[['app', 'is_attributed']].groupby('app', as_index=False).count().sort_values('is_attributed', ascending=False)
    merge = counts.merge(proportion, on='app', how='left')
    merge.columns = ['app', 'click_count', 'prop_downloaded']

    ax = merge[:100].plot(secondary_y='prop_downloaded')
    plt.title('Conversion Rates over Counts of 100 Most Popular Apps')
    ax.set(ylabel='Count of clicks')
    plt.ylabel('Proportion Downloaded')
    plt.show()

    print('Counversion Rates over Counts of Most Popular Apps')
    print(merge[:20])

def os_analysis(data):
    proportion = data[['os', 'is_attributed']].groupby('os', as_index=False).mean().sort_values('is_attributed', ascending=False)
    counts = data[['os', 'is_attributed']].groupby('os', as_index=False).count().sort_values('is_attributed', ascending=False)
    merge = counts.merge(proportion, on='os', how='left')
    merge.columns = ['os', 'click_count', 'prop_downloaded']

    ax = merge[:100].plot(secondary_y='prop_downloaded')
    plt.title('Conversion Rates over Counts of 100 Most Popular Operating Systems')
    ax.set(ylabel='Count of clicks')
    plt.ylabel('Proportion Downloaded')
    plt.show()

    print('Counversion Rates over Counts of Most Popular Operating Systems')
    print(merge[:20])

def device_analysis(data):
    proportion = data[['device', 'is_attributed']].groupby('device', as_index=False).mean().sort_values('is_attributed', ascending=False)
    counts = data[['device', 'is_attributed']].groupby('device', as_index=False).count().sort_values('is_attributed', ascending=False)
    merge = counts.merge(proportion, on='device', how='left')
    merge.columns = ['device', 'click_count', 'prop_downloaded']

    ax = merge[:50].plot(secondary_y='prop_downloaded')
    plt.title('Conversion Rates over Counts of 50 Most Popular Devices')
    ax.set(ylabel='Count of clicks')
    plt.ylabel('Proportion Downloaded')
    plt.show()

    print('Count of clicks and proportion of downloads by device:')
    print(merge[:20])

def channel_analysis(data):
    proportion = data[['channel', 'is_attributed']].groupby('channel', as_index=False).mean().sort_values('is_attributed', ascending=False)
    counts = data[['channel', 'is_attributed']].groupby('channel', as_index=False).count().sort_values('is_attributed', ascending=False)
    merge = counts.merge(proportion, on='channel', how='left')
    merge.columns = ['channel', 'click_count', 'prop_downloaded']

    ax = merge[:100].plot(secondary_y='prop_downloaded')
    plt.title('Conversion Rates over Counts of 100 Most Popular Channels')
    ax.set(ylabel='Count of clicks')
    plt.ylabel('Proportion Downloaded')
    plt.show()

    print('Counversion Rates over Counts of Most Popular Channels')
    print(merge[:20])

def dailyhour_analysis(time_analyses):
    time_analyses['click_rnd']=time_analyses['click_time'].dt.round('H')   

    #check for hourly patterns
    time_analyses[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).count().plot()
    plt.title('12 HOUR CLICK FREQUENCY');
    plt.ylabel('Number of Clicks');

    time_analyses[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).mean().plot()
    plt.title('12 HOUR CONVERSION RATIO');
    plt.ylabel('Converted Ratio');['click_time']

def hour_analysis(time_analyses):
    time_analyses[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).count().plot(kind='bar', color='#a675a1')
    plt.title('HOURLY CLICK FREQUENCY Barplot');
    plt.ylabel('Number of Clicks');

    time_analyses[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).count().plot(color='#a675a1')
    plt.title('HOURLY CLICK FREQUENCY Lineplot');
    plt.ylabel('Number of Clicks');

def load_data():
    training_data = pd.read_csv("https://raw.githubusercontent.com/patelr4/TalkingData/master/train.csv", nrows=70000)
    test_data = pd.read_csv("https://raw.githubusercontent.com/patelr4/TalkingData/master/train.csv", skiprows=range(1, 70000), nrows=30000)
    return(training_data, test_data)
def data_cleaning(data):
    del data['attributed_time'] #blank column
    data['click_time'] =  dd.to_datetime(data['click_time']) #convert from int to datetime object
    data['day'] = data['click_time'].dt.day #create new variable day
    data['hour'] = data['click_time'].dt.hour #create new variable hour
    del data['click_time']
    data.columns = ['ip', 'app', 'device', 'os','channel','is_attributed','day','hour']

    data.astype(set_data_type)
    return data

def analysing_data(data):
    nrows = len(data)
    print("Number of rows in the dataset: ", nrows)
    npositive = data.is_attributed.sum() #since is_attributed has either 0 or 1. 1 is for positive cases
    print("Number of positive cases are " + str(npositive))
    nnegative = nrows - npositive
    positive_ratio = np.longdouble(npositive/nrows)
    print("Positive data ratio is ", positive_ratio*100, "%")

def data_balancing(data):
    no_rows = len(data)
    no_pos_rows = data.is_attributed.sum()
    no_neg_rows = no_rows - no_pos_rows
    ratio = np.longdouble(no_pos_rows/no_rows)
    postive_set = data[(data['is_attributed'] == 1)]
    random_int = random.randint(1,50)
    random_state = np.random.RandomState(random_int)
    negative_set =  data[(data['is_attributed'] == 0)].sample(frac=ratio, random_state=random_state)
    temp_list = [postive_set, negative_set]
    balanced_set = pd.concat(temp_list)
    return balanced_set

def draw_roc(model, data):
    train = pd.DataFrame()
    validation = pd.DataFrame()
    #building data for crossvalidation
    random_int = random.randint(1,10)
    random_state = np.random.RandomState(random_int)
    validation_split_size = 0.3
    train, validation = model_tts(data, test_size=validation_split_size, random_state=random_state, shuffle=True )
    #Get X and y
    y_train = train['is_attributed']
    x_train = train.drop('is_attributed',axis=1)
    y_validation = validation['is_attributed']
    x_validation = validation.drop('is_attributed',axis=1)
    tprs = []
    aucs = []
    result_dict = {}
    mean_fpr = np.linspace(0, 1, 100)
    trained_model = model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_validation)
    fpr, tpr, thresholds = roc_curve(y_validation, probabilities[:, 1])    
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    result_dict["model"] = model
    result_dict["fpr"] = fpr
    result_dict["tpr"] = tpr
    result_dict["lw"] = 1
    result_dict["alpha"] = 0.3
    result_dict["roc_fold"] = 0
    result_dict["roc_auc"] = roc_auc
    return result_dict

def get_roc_test(model, test_data):
    y_test = test_data['is_attributed']
    x_test = test_data.drop('is_attributed',axis=1)
    tprs = []
    aucs = []
    result_dict = {}
    mean_fpr = np.linspace(0, 1, 100)
    probabilities = model.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    result_dict["model"] = model
    result_dict["fpr"] = fpr
    result_dict["tpr"] = tpr
    result_dict["lw"] = 1
    result_dict["alpha"] = 0.3
    result_dict["roc_fold"] = 0
    result_dict["roc_auc"] = roc_auc
    return result_dict
