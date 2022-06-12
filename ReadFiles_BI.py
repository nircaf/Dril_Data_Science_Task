import pandas as pd
import numpy as np
import datetime
from meteostat import Point, Daily
from geopy.geocoders import Nominatim
from alive_progress import alive_bar

# geopy
# Initialize Nominatim API
geolocator = Nominatim(user_agent="MyApp")

# Const-parameters
trends_country = 'IL'
category_get_only = 'דגים/בשר משומר'
branch_rishon_only = 'ראשון מזרח'
with_plot = 0
mean_dates_to_slope = 3
Delta_days_weather = 2


def get_trends_features(date, kw_list=["War", 'Recall']):  # Get trends features
    ## Trends
    # connect to google
    from pytrends.request import TrendReq
    delta_days_trends = 14

    pytrends = TrendReq(hl='en-US', tz=360)
    # build payload
    keywords = pytrends.suggestions(keyword=kw_list[0])

    pytrends.build_payload(kw_list, cat=0,
                           timeframe=(date - datetime.timedelta(days=delta_days_trends)).strftime("%Y-%m-%d") + ' '
                                     + ((date).strftime("%Y-%m-%d")),
                           geo=trends_country)
    # 1 Interest over Time
    data = pytrends.interest_over_time()
    data = data.reset_index()
    trend_slope_war = np.mean(
        np.diff(data[kw_list[0]].to_numpy(), axis=0)[delta_days_trends - mean_dates_to_slope:delta_days_trends], axis=0)
    trend_slope_Recall = np.mean(
        np.diff(data[kw_list[1]].to_numpy(), axis=0)[delta_days_trends - mean_dates_to_slope:delta_days_trends], axis=0)

    if with_plot:
        data.plot(kind='line', x="date", y=kw_list)

    return trend_slope_war, trend_slope_Recall


def load_all_data(filepath):
    return pd.read_csv(filepath)


def Get_all_data(csv_files, savefile=True):  # get all the BI data
    all_data = pd.DataFrame()
    all_labels = pd.Series(dtype=int)
    with alive_bar(len(csv_files), force_tty=True) as bar:
        for csv_file in csv_files:
            col_to_add = pd.DataFrame(columns=['tempavg', 'precipitation', 'wind_speed', "War", 'Recall'])
            bar()
            df = pd.read_excel(csv_file)
            df = df[df.Pidyon >= 0]  # Get only non-negative pidyon
            df = df[df.category == category_get_only]  # Get only specific cetegory
            df = df.reset_index()
            if len(df) == 0:
                continue
            # Get date and google trends features
            date = df.date[0]
            trend_slope_war, trend_slope_Recall = get_trends_features(date)
            location = geolocator.geocode('Tel Aviv')  # Get location latitude longitude
            location_geolocator = Point(location.latitude, location.longitude)  # get geolocator location
            get_weather = Daily(location_geolocator, date - datetime.timedelta(days=Delta_days_weather), date)
            get_weather_except = get_weather.fetch()
            for index, content in df.iterrows():
                # date.weekday()
                branch_location = content.branch.split('-')[-1]
                try:
                    # Weather API
                    location = geolocator.geocode(branch_location)  # Get location latitude longitude
                    location_geolocator = Point(location.latitude, location.longitude)  # get geolocator location
                    get_weather = Daily(location_geolocator, date - datetime.timedelta(days=Delta_days_weather), date)
                    get_weather = get_weather.fetch()
                    if len(get_weather.tavg.values) == 0: # If location found but no data
                        to_except() # Go to except
                except:
                    print(
                        "branch location: " + branch_location + " was not found")  # This exception should be fixed if we use GoogleMaps
                    get_weather = get_weather_except
                # Trends API
                col_to_add = pd.concat([col_to_add, pd.DataFrame.from_records([{'tempavg':
                                                                                    np.mean(
                                                                                        get_weather.tavg.values),
                                                                                'precipitation': np.mean(
                                                                                    get_weather.prcp.values),
                                                                                'wind_speed':
                                                                                    np.mean(
                                                                                        get_weather.wspd.values),
                                                                                "War": trend_slope_war,
                                                                                'Recall': trend_slope_Recall}])],
                                       ignore_index=True)
            all_labels = pd.concat([all_labels, df.Pidyon])
            all_labels.reset_index(drop=True, inplace=True)
            if {'Number','Price'}.issubset(df.columns):
                df.drop(['index', 'barCode', 'Pidyon','Number','Price'], axis=1, inplace=True)
            else:
                df.drop(['index', 'barCode', 'Pidyon'], axis=1, inplace=True)
            df.reset_index(drop=True, inplace=True)
            col_to_add.reset_index(drop=True, inplace=True)
            x_data = pd.concat([df, col_to_add], axis=1)
            x_data.reset_index(drop=True, inplace=True)
            all_data = pd.concat([all_data, x_data], ignore_index=True)
            all_data.reset_index(drop=True, inplace=True)
    if savefile:
        all_data.to_csv('saved_data/all_data.csv')
        all_labels.to_csv('saved_data/all_labels.csv')
        print("all_data saved")


    return all_data, all_labels

def Get_all_Rishon(csv_files, savefile=True):  # get all the BI data
    all_data = pd.DataFrame()
    all_labels = pd.Series(dtype=int)
    with alive_bar(len(csv_files), force_tty=True) as bar:
        for csv_file in csv_files:
            bar()
            col_to_add = pd.DataFrame(columns=['tempavg', 'precipitation', 'wind_speed', "War", 'Recall'])
            df = pd.read_excel(csv_file)
            df = df[df.Pidyon >= 0]  # Get only non-negative pidyon
            df = df[df.category == category_get_only]  # Get only specific cetegory
            df = df[df.branch.str.contains(branch_rishon_only)] # Get only specific cetegory
            df = df.reset_index()
            if len(df) ==  0:
                continue
            # Get date and google trends features
            date = df.date[0]
            trend_slope_war, trend_slope_Recall = get_trends_features(date)
            location = geolocator.geocode('Tel Aviv')  # Get location latitude longitude
            location_geolocator = Point(location.latitude, location.longitude)  # get geolocator location
            get_weather = Daily(location_geolocator, date - datetime.timedelta(days=Delta_days_weather), date)
            get_weather_except = get_weather.fetch()
            for index, content in df.iterrows():
                # date.weekday()
                branch_location = content.branch.split('-')[-1]
                try:
                    # Weather API
                    location = geolocator.geocode(branch_location)  # Get location latitude longitude
                    location_geolocator = Point(location.latitude, location.longitude)  # get geolocator location
                    get_weather = Daily(location_geolocator, date - datetime.timedelta(days=Delta_days_weather), date)
                    get_weather = get_weather.fetch()
                    if len(get_weather.tavg.values) == 0: # If location found but no data
                        to_except() # Go to except
                except:
                    print(
                        "branch location: " + branch_location + " was not found")  # This exception should be fixed if we use GoogleMaps
                    get_weather = get_weather_except
                # Trends API
                col_to_add = pd.concat([col_to_add, pd.DataFrame.from_records([{'tempavg':
                                                                                np.mean(
                                                                                    get_weather.tavg.values),
                                                                            'precipitation': np.mean(
                                                                                get_weather.prcp.values),
                                                                            'wind_speed':
                                                                                np.mean(
                                                                                    get_weather.wspd.values),
                                                                            "War": trend_slope_war,
                                                                            'Recall': trend_slope_Recall}])],
                                   ignore_index=True)
            # Finished going over an excel ^^
            all_labels = pd.concat([all_labels, df.Pidyon])
            all_labels.reset_index(drop=True, inplace=True)
            if {'Number','Price'}.issubset(df.columns): # Iכ
                print(df.columns)
                df.drop(['index', 'barCode', 'Pidyon','Number','Price'], axis=1, inplace=True)
            else:
                df.drop(['index', 'barCode', 'Pidyon'], axis=1, inplace=True)
            df.reset_index(drop=True, inplace=True)
            col_to_add.reset_index(drop=True, inplace=True)
            x_data = pd.concat([df, col_to_add], axis=1)
            x_data.reset_index(drop=True, inplace=True)
            all_data = pd.concat([all_data, x_data], ignore_index=True)
            all_data.reset_index(drop=True, inplace=True)
    if savefile:
        all_data.to_csv('saved_data/all_data_Rishon.csv')
        all_labels.to_csv('saved_data/all_labels_Rishon.csv')


    return all_data, all_labels
