import pandas as pd
import ReadFiles_BI
import numpy as np
import tabular_DL_pytorch_regression
import ReadFiles_CV
import Kmeans_CV
from os import listdir
from os.path import isfile, join
import time
import tabular_DL_pytorch_regression_tuning
import model_sklearn

# get dir excels
mypath = 'BI'  # location of BI files
csv_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]  # Get BI files


def merge_BI_CV(all_data, split_to_dates,all_labels):
    for row in split_to_dates.iterrows():
        all_data.loc[all_data.date == row[0], row[1].axes[0].values] = row[1].values
    all_data = all_data.iloc[:, 1:]
    all_labels = all_labels.drop(np.where(all_data.isnull().any(axis=1))[0])
    return all_data.dropna(axis=0, how='any'),all_labels  # remove unmatching rows


def CV_Run():
    engagement_percent_mean_loc = ReadFiles_CV.get_precent_engagement_w_location()
    Kmeans_CV.kmeans_run(engagement_percent_mean_loc)


def BI_Run():
    print('BI all data')
    # all_data, all_labels = ReadFiles_BI.Get_all_data(csv_files)  # Run all data. if one run [csv_files[0]]
    all_data = ReadFiles_BI.load_all_data('saved_data/all_data.csv')  # Load data
    all_labels = ReadFiles_BI.load_all_data('saved_data/all_labels.csv').iloc[:, 1]  # Load data
    all_data_ranked = Kmeans_CV.excels_to_num(all_data)  # factorize data

    # model_sklearn.model_run(all_data_ranked,all_labels)
    if tuning:
        tabular_DL_pytorch_regression_tuning.tabular_DL_torch(all_labels, all_data_ranked, True)
    else:
        tabular_DL_pytorch_regression.tabular_DL_torch(all_labels, all_data_ranked)  # Build model to learn data


def Combined_Run():
    # combined BI CV
    split_to_dates = ReadFiles_CV.get_precent_engagement_w_location_to_dates()  # Read CV files and get metric
    # all_data, all_labels = ReadFiles_BI.Get_all_Rishon(csv_files, savefile=True) # Get data from excel

    all_data = ReadFiles_BI.load_all_data('saved_data/all_data_Rishon.csv')  # Load data
    all_labels = ReadFiles_BI.load_all_data('saved_data/all_labels_Rishon.csv').iloc[:, 1]  # Load data

    all_data,all_labels = merge_BI_CV(all_data, split_to_dates,all_labels)
    all_data.reset_index(drop=True, inplace=True)
    all_labels.reset_index(drop=True, inplace=True)
    all_data = Kmeans_CV.excels_to_num(all_data)  # factorize data

    if tuning:
        tabular_DL_pytorch_regression_tuning.tabular_DL_torch(all_labels, all_data, BI_Data=False)
    else:
        tabular_DL_pytorch_regression.tabular_DL_torch(all_labels, all_data,use_best_hyperparameters = True,BI_Data=False)  # Build model to learn data


if __name__ == '__main__':
    tic = time.time()
    tuning = False
    # CV_Run()
    # BI_Run()
    Combined_Run()
    print("Runtime {0}".format(time.time() - tic))
