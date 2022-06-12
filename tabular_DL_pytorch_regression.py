import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from pytorch_tabnet.tab_model import TabNetRegressor
import json
from sklearn.model_selection import train_test_split
from scipy import stats
from alive_progress import alive_bar

def load_best_results(BI_Data): # Load config to tabnet
    with open('saved_data/best_result_df.json' if BI_Data else 'saved_data/best_result_df_combined.json') as f:
        return json.load(f)


def tabular_DL_torch(target, features,use_best_hyperparameters = True,BI_Data=True):
    print('tabular_DL_torch')
    train = features.copy()
    train = train.to_numpy()
    target = target.to_numpy().reshape(-1, 1)
    if use_best_hyperparameters:
        config = load_best_results(BI_Data)
        TabNetArgs = {'n_d':int(config['n_d']), 'n_a':int(config['n_d']), 'n_steps':int(config['n_steps']), 'gamma':1.3,
                     'lambda_sparse':0, 'momentum':config['momentum'], 'optimizer_fn':torch.optim.Adam,
                     'optimizer_params': {'lr':config['lr'], 'weight_decay':config['weight_decay']},
                     'mask_type':'entmax',
                     'scheduler_params':{'mode':"min",
                                           'patience':5,
                                           'min_lr':1e-5,
                                           'factor':0.9,},
                     'scheduler_fn':torch.optim.lr_scheduler.ReduceLROnPlateau,
                     'verbose':100,
                     }
    else:
        TabNetArgs = {'n_d':24, 'n_a':24, 'n_steps':4, 'gamma':1.3,
                     'lambda_sparse':0, 'momentum':0.02, 'optimizer_fn':torch.optim.Adam,
                     'optimizer_params': {'lr':2e-2, 'weight_decay':1e-5},
                     'mask_type':'entmax',
                     'scheduler_params':{'mode':"min",
                                           'patience':5,
                                           'min_lr':1e-5,
                                           'factor':0.9,},
                     'scheduler_fn':torch.optim.lr_scheduler.ReduceLROnPlateau,
                     'verbose':0,
                     }
    kf = KFold(n_splits=5, random_state=42, shuffle=True) # increase n_splits to incrase accuracy and timerun
    train, x_test, target, y_test = train_test_split(train, target, test_size=0.20)
    CV_score_array = []
    max_epochs = 1000 # Increase epoch number to incrase accuracy and timerun
    with alive_bar(kf.n_splits, force_tty=True) as bar:
        for train_index, test_index in kf.split(train):
            x_train, x_valid = train[train_index], train[test_index]
            y_train, y_valid = target[train_index], target[test_index]
            clf = TabNetRegressor(**TabNetArgs)
            clf.fit(X_train=x_train, y_train=y_train,
                    eval_set=[(x_valid, y_valid)],
                    patience=max_epochs / 5, max_epochs=max_epochs,
                    eval_metric=['rmse'])
            CV_score_array.append(clf.best_cost)
            bar()
    print("The CV score is %.5f" % np.mean(CV_score_array, axis=0))

    # plot losses
    plt.plot(clf.history['val_0_rmse'][5:], label='validation rmse')
    plt.xlabel('Epochs')
    plt.ylabel('rmse')
    plt.show(block=False)
    print('RMSE ttest result: {0}'.format((stats.ttest_ind(clf.predict(x_test), y_test)[0][0])))
    print('RMSE ttest pvalue: {0}'.format((stats.ttest_ind(clf.predict(x_test), y_test)[1][0])))

