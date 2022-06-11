
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import isnan
from sklearn.impute import KNNImputer

import pytorch_tabnet
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score , log_loss
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import train_test_split
from scipy import stats
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

class my_metric(Metric):
    """
    2xAUC.
    """

    def __init__(self):
        self._name = "custom" # write an understandable name here
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            AUC of predictions vs targets.
        """
        return 2*roc_auc_score(y_true, y_score[:, 1])

def tabular_DL_torch(target,features):
    targetbu = target
    featuresbu = features
    target2 = np.sqrt(stats.zscore(np.array(target)).astype(int)).astype(int)
    # target2 = stats.zscore(np.array(target)).astype(int).astype(int)
    target3 = np.zeros((np.shape(target2)[0],max(target2)+1))
    for i in np.unique(target2):
        target3[:,i] += (target2==i).astype(int)
        # target3 = np.concatenate((target3, (target==i).astype(int)),axis = 1)

    # Encoding train set
    # test = pd.read_csv('TabNetMultiTaskClassifier_baseline/test_features.csv')
    train = featuresbu
    # test = featuresbu
    np.random.seed(42)
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid"], p=[.8, .2], size=(train.shape[0],))
    train_indices = train[train.Set == 0].index
    valid_indices = train[train.Set == 1].index
    nunique = train.nunique()
    types = train.dtypes
    categorical_columns = []
    categorical_dims = {}
    for col in train.columns:   #tqdm(train.columns):
        if types[col] == 'object' or nunique[col] < 200:
            print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            training_mean = train.loc[train_indices, col].mean()
            train.fillna(training_mean, inplace=True)

    unused_feat = ['Set']  # Let's not use splitting sets and sig_id

    features = [col for col in train.columns if col not in unused_feat]

    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    x_train = train[features].values[train_indices]
    y_train = target[train_indices]

    x_valid = train[features].values[valid_indices]
    y_valid = target[valid_indices]


    # define the model
    clf = TabNetMultiTaskClassifier(n_steps=1,
                                    cat_idxs=cat_idxs,
                                    cat_dims=cat_dims,
                                    cat_emb_dim=1,
                                    optimizer_fn=torch.optim.Adam,
                                    optimizer_params=dict(lr=2e-2),
                                    scheduler_params={"step_size": 50,  # how to use learning rate scheduler
                                                      "gamma": 0.9},
                                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                    mask_type='entmax',  # "sparsemax",
                                    lambda_sparse=0,  # don't penalize for sparser attention
                                    )

    max_epochs = 10 # 1000
    clf.fit(
        X_train=x_train, y_train=y_train,
        max_epochs=max_epochs,
        patience=50,  # please be patient ^^
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=1,
        drop_last=False,
    )

    # plot losses
    plt.plot(clf.history['loss'])
    # plot auc
    plt.plot(clf.history['train_auc'])
    plt.plot(clf.history['val_auc'])
    # plot learning rates
    plt.plot([x for x in clf.history['lr']])
    preds_valid = clf.predict_proba(x_valid)  # This is a list of results for each task

    # We are here getting rid of tasks where only 0 are available in the validation set
    valid_aucs = [roc_auc_score(y_score=task_pred[:, 1], y_true=y_valid[:, task_idx])
                  for task_idx, (task_pred, n_pos) in enumerate(zip(preds_valid, y_valid.sum(axis=0))) if n_pos > 0]

    valid_logloss = [log_loss(y_pred=task_pred[:, 1], y_true=y_valid[:, task_idx])
                     for task_idx, (task_pred, n_pos) in enumerate(zip(preds_valid, y_valid.sum(axis=0))) if n_pos > 0]

    plt.scatter(y_valid.sum(axis=0)[y_valid.sum(axis=0) > 0], valid_aucs)

    # plot accuracy
    plt.plot(clf.history['train_accuracy'])
    plt.plot(clf.history['valid_accuracy'])
    #
    # find and plot feature importance
    y_pred = clf.predict(x_valid)
    clf.feature_importances_
    feat_importances = pd.Series(clf.feature_importances_, index=features.columns)
    feat_importances.nlargest(20).plot(kind='barh')
    #
    #
    # determine best accuracy for validation set
    preds_valid = clf.predict(x_valid)
    valid_acc = accuracy_score(preds_valid, y_valid)

    # print(f"BEST ACCURACY SCORE ON VALIDATION SET : {valid_acc}")
    # print(f"BEST ACCURACY SCORE ON TEST SET : {test_acc}")

