import matplotlib.pyplot as plt

def plott(min_features_to_select,rfecv):
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
        rfecv.grid_scores_,
    )
    plt.show()

def barr(importance,feature_names,ytitle,titles,save_name):
    plt.figure()
    plt.bar(height=importance, x=feature_names)
    plt.title(titles)
    plt.ylabel(ytitle)
    plt.show(block=False)
    plt.savefig(save_name+'.png')