import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from pytorch_tabnet.tab_model import TabNetRegressor
import ray
from ray import tune

# tensorboard --logdir logs/fit

def tabular_DL_torch(all_labels, all_data, BI_Data=True):
    train = all_data.copy()
    target = all_labels.copy()
    train = train.to_numpy()
    target = target.to_numpy().reshape(-1, 1)
    kf = KFold(n_splits=2, random_state=42, shuffle=True)
    ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
    ray.init(log_to_driver=False, local_mode=True)

    def train_model(config):
        print('enter_train_model')
        max_epochs = 100
        CV_score_array = []
        for train_index, test_index in kf.split(train):
            x_train, x_valid = train[train_index], train[test_index]
            y_train, y_valid = target[train_index], target[test_index]
            clf = TabNetRegressor(n_d=config['n_d'], n_a=config['n_d'], n_steps=config['n_steps'],
                                  momentum=config['momentum'], optimizer_fn=torch.optim.Adam, verbose = max_epochs/2,
                                  optimizer_params=dict(lr=config['lr'], weight_decay=config['weight_decay']), seed=42)
            clf.fit(X_train=x_train, y_train=y_train,
                    eval_set=[(x_valid, y_valid)],
                    patience=max_epochs / 5, max_epochs=max_epochs,
                    eval_metric=['rmse'])
            CV_score_array.append(clf.best_cost)
            print('b')
        print('c')
        print("The CV score is %.5f" % np.mean(CV_score_array, axis=0))
        return dict(rmse=np.mean(CV_score_array, axis=0))

    analysis = tune.run(
        train_model,
        config={"lr": tune.grid_search([1e-2, 1e-3, 1e-4]),
                "momentum": tune.uniform(0.01, 0.1),
                "n_d": tune.grid_search([8, 24, 32]),
                "weight_decay": tune.uniform(0, 1e-3),
                "n_steps": tune.randint(1, 10)},
        resources_per_trial={"cpu": 8, "gpu": 1},
        verbose=3,
        name="train_model",  # This is used to specify the logging directory.
        num_samples=2,
        metric="rmse",
        mode="min",
        stop={"rmse": 100}, max_failures=0  # This will stop the trial
        # , keep_checkpoints_num=1, checkpoint_score_attr="val_0_rmse"
    )
    print("best config: ", analysis.get_best_config(metric="rmse", mode="min"))
    pd.Series(analysis.get_best_config(metric="rmse", mode="min")).to_json(
        'saved_data/best_result_df.json' if BI_Data else 'saved_data/best_result_df_combined.json')


def excels_to_num(data):  # factorize data
    for column in data:
        data[column] = pd.factorize(data[column])[0]


if __name__ == '__main__':
    import ReadFiles_BI
    from os import listdir
    from os.path import isfile, join

    mypath = 'BI'
    csv_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    features, target = ReadFiles_BI.Get_all_Rishon(csv_files)
    # features, target = ReadFiles_BI.Get_all_data(csv_files)
    excels_to_num(features)
    tabular_DL_torch(target, features)
