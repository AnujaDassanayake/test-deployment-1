import os
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta




def data_for_time_window_mlr(SCADA_dataframe, start_date, end_date, target_column, features_to_remove):

  all_features = list(SCADA_dataframe.columns)
  features_to_remove.append(target_column)
  for i in features_to_remove:
    all_features.remove(i)  

  SCADA_dataframe_date_range = SCADA_dataframe[start_date:end_date]

  # print(start_date + ' to ' + end_date + str(len(SCADA_dataframe_date_range)))

  #check if the number of records available within the test window is > 0
  #if >0 create test window
  #if <0 do not create test window
  if len(SCADA_dataframe_date_range) > 0:
    X_test1 = SCADA_dataframe_date_range[all_features]
    y_test1 = SCADA_dataframe_date_range[target_column]
    time = start_date
    status = 'good'
    return X_test1, y_test1, time, status
  else:
    status = 'bad'
    X_test1 , y_test1 = 0, 0
    time = start_date
    return X_test1, y_test1, time, status
  

def evaluate_model(model, X_test, y_test, time_period): 
  #Predicting dependent variable when independent variables are given as inputs
  test_predictions = model.predict(X_test)

  mae = tf.keras.metrics.MeanAbsoluteError()
  mse = tf.keras.metrics.MeanSquaredError()
  rmse = tf.keras.metrics.RootMeanSquaredError()
  mape = tf.keras.metrics.MeanAbsolutePercentageError()

  #Error matrics calculation
  mae.update_state(y_test,test_predictions)
  mse.update_state(y_test,test_predictions)
  rmse.update_state(y_test,test_predictions)
  mape.update_state(y_test,test_predictions)

  mae_out = mae.result().numpy()
  mse_out = mse.result().numpy()  
  rmse_out = rmse.result().numpy()
  mape_out = mape.result().numpy()
  r2_out = model.score(X_test,y_test)
  #Get the number of rows in the test window
  row_count = len(X_test)

  return {'Time_Period':time_period, 'rows':row_count, 'MAE':mae_out, 'MSE':mse_out ,'RMSE':rmse_out, 'MAPE':mape_out, 'R2':r2_out}


def calculate_error_matrices( dataset, model,target_col, features_to_remove):
  """
  This function creates evaluation df
  """

  start_date = str(dataset.index.min().date())
  print('start date ' + start_date)
  end_date = str(dataset.index.max().date())
  print('end date ' + end_date)
  lag = 1 #creates daily windows
  expected_recs = 6*24*lag #number of expected records per day
  current_date = datetime.strptime(start_date, '%Y-%m-%d')
  last_date = datetime.strptime(end_date, '%Y-%m-%d')
  current_date_plus_lag = current_date + timedelta(days=(lag-1))  
  eval_data = pd.DataFrame(columns=['Time_Period','rows','MAE','MSE','RMSE','MAPE','R2'])
  i=0
  while current_date < last_date:
    window_start_date = str(current_date.date())
    window_end_date = str(current_date_plus_lag.date())

    features_remove = features_to_remove.copy()

    X_test1, y_test1, time, status = data_for_time_window_mlr( dataset, window_start_date, window_end_date, target_col, features_remove)
    #only add rows to the test window dataframe if status is good
    #status good when number of records for the test window is > 0
    if status == 'good':
      #adding row with evaluation metrices to the dataframe
      eval_data.loc[i] = pd.Series(evaluate_model(model,X_test1,y_test1,time))
    #moving to the next timestep
    current_date_plus_lag = current_date_plus_lag + timedelta(days=lag)  
    current_date = current_date + timedelta(days=lag) 
    i+=1
    eval_data['completeness'] = eval_data['rows']/expected_recs
    
  return eval_data
