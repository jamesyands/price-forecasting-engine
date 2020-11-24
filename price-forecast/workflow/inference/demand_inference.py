import argparse
import pickle
from pathlib import Path
import pymysql
import sys
import datetime
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine


def time_add_date(df):
    return (df['DATE_PREDICT'] + \
            timedelta(days=df['PERIOD'] // 49)) #.strftime("%d/%m/%Y")

def period_to_datetime(df):
    return (df['DATE_PREDICT'] + \
            timedelta(hours=df['PERIOD'] * 0.5)) #.strftime("%d/%m/%Y %H:%M:%S")


def output_data_format(y_pred, type):
    if type == 'demand':
        output_df = pd.DataFrame({'DEMAND_FORECAST': y_pred,
                                  'PERIOD': np.arange(1, 49)})
    elif type == 'usep':
        output_df = pd.DataFrame({'USEP_FORECAST': y_pred,
                                  'PERIOD': np.arange(1, 49)})
    output_df['TIME_INSERT'] = datetime.now(pytz.timezone('Asia/Singapore'))
    output_df['PERIOD'] = int(datetime.now(pytz.timezone('Asia/Singapore')
                                          ).strftime("%d/%m/%Y %H:%M:%S")[11:13]) * 2 + \
                          int(datetime.now(pytz.timezone('Asia/Singapore')
                                          ).strftime("%d/%m/%Y %H:%M:%S")[14:16]) // 30 + \
                          output_df['PERIOD'] # + 1
    output_df['DATE_PREDICT'] = datetime.now(pytz.timezone('Asia/Singapore')).replace(
        minute=0, hour=0, second=0, microsecond=0)
    output_df['DATE_PREDICT'] = output_df.apply(time_add_date, axis=1)
    output_df['PERIOD'] = output_df['PERIOD'].apply(lambda x: x if x < 49 else x - 48)
    output_df['TIME_PREDICT'] = output_df.apply(period_to_datetime, axis=1)
    output_df['TIME_INSERT'] = pd.to_datetime(output_df['TIME_INSERT'])
    if type == 'demand':
        output_df = output_df[['TIME_INSERT', #'DATE_PREDICT', 'PERIOD',
                               'TIME_PREDICT', 'DEMAND_FORECAST']]
    elif type == 'usep':
        output_df = output_df[['TIME_INSERT', #'DATE_PREDICT', 'PERIOD',
                               'TIME_PREDICT','USEP_FORECAST']]
    return output_df


def mape_cal(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def main(args):
    parser = argparse.ArgumentParser(description='usep forecast inference params')
    parser.add_argument("--demand_model_filename", type=str, required=True)
    parser.add_argument("--usep_model_filename", type=str, required=True)
    parser.add_argument("--pb_model_filename", type=str, required=True)
    parser.add_argument("--test_start", type=str, required=True)
    parser.add_argument("--test_end", type=str, required=True)
    parser.add_argument("--usep_mode_service_url", type=str, required=True)
    args = parser.parse_args(args)

    engine = create_engine('mysql+pymysql://appuser:Appuser123456!@10.65.19.35:3306/sgms_forecast', echo=False)

    # demand_model_filename = 'demand_forecast_model_rf.sav'
    demand_model_filename = args.demand_model_filename
    # usep_model_filename = 'usep_forecast_model_rf.sav'
    usep_model_filename = args.usep_model_filename
    # test_start, test_end = '2018-12-01', '2018-12-02'
    test_start, test_end = args.test_start, args.test_end

    ##############################  Demand Forecast  #############################
    # Read Demand Data From MySQL
    query = '''
    SELECT * FROM demand_feature
    '''
    with engine.connect() as connection:
        demand_df = pd.read_sql_query(query, engine).sort_values(['DATE', 'PERIOD'])

    # Processing Load Forecast Data
    feature_normalized = demand_df.drop(['DATE'], axis=1)
    feature_normalized['DEMAND_FORECAST'] = feature_normalized['DEMAND'].shift(-1).ffill()
    testset_filter = (demand_df['DATE'] >= test_start) & (demand_df['DATE'] < test_end)
    feature_list = list(feature_normalized.drop(['DEMAND_FORECAST'], axis=1).columns)
    X_test, y_test = feature_normalized[feature_list][testset_filter].to_numpy(), \
                     feature_normalized['DEMAND_FORECAST'][testset_filter].to_numpy()
    print(X_test.shape, y_test.shape)
    # Load demand model from Git
    loaded_model = pickle.load(open(demand_model_filename, 'rb'))
    print(demand_model_filename)
    y_pred = loaded_model.predict(X_test)
    mape = mape_cal(y_pred, y_test)
    print(mape)
    print(y_pred)

    # Write Load Forecast result to output node and MySQL
    Path(args.demand_forecast_result).parent.mkdir(parents=True, exist_ok=True)
    with open(args.demand_forecast_result, 'w') as output_path:
        output_path.write(str(list(y_pred))[1:-1])
    output_data_format(y_pred,'demand').to_sql('demand_forecast_result', con=engine, if_exists='append', index=False)


    ##############################  USEP Forecast  #############################
    # Read USEP Data From MySQL
    query = '''
    SELECT * FROM usep_feature LIMIT 35040
    '''
    with engine.connect() as connection:
        usep_df = pd.read_sql_query(query, engine).sort_values(['DATE', 'PERIOD'])

    # Processing Load Forecast Data
    feature_normalized = usep_df.drop(['DATE'], axis=1)
    feature_normalized['USEP_FORECAST'] = feature_normalized['USEP'].shift(-1).ffill()
    testset_filter = (usep_df['DATE'] >= test_start) & (usep_df['DATE'] < test_end)
    feature_list = list(feature_normalized.drop(['USEP_FORECAST'], axis=1).columns)
    X_test, y_test = feature_normalized[feature_list][testset_filter].to_numpy(), \
                     feature_normalized['USEP_FORECAST'][testset_filter].to_numpy()
    print(X_test.shape, y_test.shape)
    # Load demand model from Git
    loaded_model = pickle.load(open(usep_model_filename, 'rb'))
    print(usep_model_filename)
    y_pred = loaded_model.predict(X_test)
    mape = mape_cal(y_pred, y_test)
    print(mape)
    print(y_pred)

    output_data_format(y_pred,'usep').to_sql('usep_forecast_backup', con=engine, if_exists='append', index=False)

if __name__ == '__main__':
    main(sys.argv[1:])
