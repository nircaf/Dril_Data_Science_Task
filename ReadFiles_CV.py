import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
import warnings
import datetime

with open('CV/actions_tuna.json') as f: # Get CV Data
    data = json.load(f)
# Categorize actions
encoding_dictionary = {'hold_product': 1, 'pick_up': 2, 'put_back': 6, 'put_to_cart': 3, 'touch_product': 4,
                       'stop_at_category': 5, 'stand': 7, 'walk': 8}

score_threshold = 2 / encoding_dictionary.__len__()
put_to_cart = 'put_to_cart'


def get_precent_engagement_w_location_to_dates(): # Return engagement metric with dates
    engagement_percent_mean_loc = []
    for index, id_costumer in enumerate(data):
        actions_above_thresh = [d['name'] for d in id_costumer["actions"] if d['score'] > score_threshold]
        engagement_with_product = [
            d['x1'] + d['x2'] + (np.sqrt(np.power(d['x1'] - d['x2'], 2) + np.power(d['y1'] - d['y2'], 2)) / 2) for d in
            id_costumer["actions"]
            if d['score'] > score_threshold and encoding_dictionary[
                d['name']] < 5]  # Get center of camera for actions of engagement
        no_engagement = [
            d['x1'] + d['x2'] + (np.sqrt(np.power(d['x1'] - d['x2'], 2) + np.power(d['y1'] - d['y2'], 2)) / 2) for d in
            id_costumer["actions"]
            if d['score'] > score_threshold and encoding_dictionary[
                d['name']] > 5]  # Get center of camera for actions of no engagement
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if engagement_with_product.__len__() > 0 and no_engagement.__len__() > 0 and len(id_costumer) == 6:
                engagement_percent_mean_loc.append({'date': id_costumer['buyer_id'].split('_')[0],
                                                    'Percent_engage': engagement_with_product.__len__() / actions_above_thresh.__len__()
                                                       ,
                                                    'location_engage': np.array(engagement_with_product).mean(),
                                                    'location_no_engage': np.array(no_engagement).mean()
                                                       , 'age': (float(id_costumer['age'].split('-')[0]) + float(
                        id_costumer['age'].split('-')[1])) / 2
                                                       , 'gender': 1 if id_costumer['gender'] == 'male' else 0,
                                                    })
    return pd.DataFrame(engagement_percent_mean_loc).groupby(
        'date').mean()


def get_precent_engagement_w_location():  # Return engagement metric
    engagement_percent_mean_loc = []
    for index, id_costumer in enumerate(data):
        actions_above_thresh = [d['name'] for d in id_costumer["actions"] if d['score'] > score_threshold]
        engagement_with_product = [
            d['x1'] + d['x2'] + (np.sqrt(np.power(d['x1'] - d['x2'], 2) + np.power(d['y1'] - d['y2'], 2)) / 2) for d in
            id_costumer["actions"]
            if d['score'] > score_threshold and encoding_dictionary[
                d['name']] < 5]  # Get center of camera for actions of engagement
        no_engagement = [
            d['x1'] + d['x2'] + (np.sqrt(np.power(d['x1'] - d['x2'], 2) + np.power(d['y1'] - d['y2'], 2)) / 2) for d in
            id_costumer["actions"]
            if d['score'] > score_threshold and encoding_dictionary[
                d['name']] > 5]  # Get center of camera for actions of no engagement
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if engagement_with_product.__len__() > 0 and no_engagement.__len__() > 0 and len(id_costumer) >= 6:
                engagement_percent_mean_loc.append({'date': id_costumer['buyer_id'].split('_')[0],
                                                    'Percent_engage': engagement_with_product.__len__() / actions_above_thresh.__len__()
                                                       ,
                                                    'delta_location_engage': np.array(
                                                        engagement_with_product).mean() - np.array(no_engagement).mean()
                                                       , 'age': (float(id_costumer['age'].split('-')[0]) + float(
                        id_costumer['age'].split('-')[1])) / 2
                                                       , 'gender': 1 if id_costumer['gender'] == 'male' else 0,
                                                    'date_in_week': datetime.datetime.strptime(id_costumer['buyer_id'].split('_')[0], '%Y-%m-%d').weekday()})
    return engagement_percent_mean_loc


def actions_process_binary():  # Return engagement metric binary
    engagement_with_product = []
    for index, id_costumer in enumerate(data):
        actions_above_thresh = [d['name'] for d in id_costumer["actions"] if d['score'] > score_threshold]
        if actions_above_thresh.count(put_to_cart) == 0:
            engagement_with_product.append(0)
        else:
            engagement_counter = 0
            joined = " ".join(actions_above_thresh)
            separated = joined.split(put_to_cart)[:-1]
            if type(separated) == list:
                for pre_cart in separated:  # run over how many items to cart
                    if pre_cart != '':
                        for j, actions in enumerate(pre_cart.split()):
                            if encoding_dictionary[actions] < 5:
                                engagement_counter += 1
                            elif actions == put_to_cart:
                                engagement_with_product.append(engagement_counter)
                                engagement_counter = 0
                            elif encoding_dictionary[actions] == 6:
                                engagement_counter = 0
    return engagement_counter


def actions_process_multiclass():  # Return engagement metric multiclass
    all_lines = []
    for index, id_costumer in enumerate(data):
        actions_above_thresh = [d['name'] for d in id_costumer["actions"] if d['score'] > score_threshold]
        joined = " ".join(actions_above_thresh)
        separated = joined.split('put_to_cart')[:-1]
        separated_to_actions = np.empty(np.shape(separated))
        a = []
        for i, pre_cart in enumerate(separated):
            if pre_cart.split()[-1] == 'touch_product':
                temp_actions = []
                for j, an_action in enumerate(pre_cart.split()):
                    temp_actions.append(encoding_dictionary[an_action])
                    np.append(separated_to_actions, encoding_dictionary[an_action])
                a.insert(i, temp_actions)
        all_lines.append(a)
    df = pd.DataFrame(all_lines)
    df = df.fillna(value='')
    actions_to_num = df.apply(preprocessing.LabelEncoder().fit_transform)
    return actions_to_num
