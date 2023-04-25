import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import seaborn as sns
import time
import os
# import calplot
import plotly.express as px
import numpy
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from windrose import WindroseAxes
from datetime import date
from datetime import timedelta


from evaluation import(
    calculate_error_matrices
)

from anomaly_detection import(
    anomaly_detection_daily
)

def anomaly_graph(test_data, regression_model_path, turbine, target_col, features2rem, ocsvm_model_path, component):
    
    test_data_use = test_data.copy()
    test_data_use['Timestamp'] = pd.to_datetime(test_data_use['Timestamp'])
    test_data_use.set_index('Timestamp',inplace = True)
    selected_turbine_scada_data = test_data_use[test_data_use['Turbine_ID'] == turbine]
    selected_turbine_scada_data = selected_turbine_scada_data.sort_values(by = ['Timestamp'], ascending = True)
    selected_turbine_scada_data.dropna(inplace=True)
    dataset = selected_turbine_scada_data

    pick_read = open(regression_model_path,'rb')
    reg_model = pickle.load(pick_read)
    pick_read.close()
    eval_df = calculate_error_matrices(dataset ,reg_model ,target_col, features2rem, 7)

    test_metrics_copy_all = anomaly_detection_daily(ocsvm_model_path, eval_df)
    test_metrics_copy_all['date'] = pd.to_datetime(test_metrics_copy_all['date'])
    test_metrics_copy_all.set_index('date',inplace = True)

    analomaly_dates = (
    test_metrics_copy_all
    .query('status_text == "anomaly"')
    )
    analomaly_dates['week_number'] = analomaly_dates.index.isocalendar().week
    analomaly_dates['week_number'] = analomaly_dates['week_number'] + 1
    analomaly_dates = analomaly_dates[['status_text','week_number']]
    analomaly_dates['Start_Date'] = analomaly_dates.index.date
    analomaly_dates['End_Date'] = analomaly_dates['Start_Date'] + timedelta(days=6)

    plot_data = selected_turbine_scada_data[[target_col]]
    plot_data['week_number'] = plot_data.index.isocalendar().week
    plot_data['datetime'] = plot_data.index

    plot_data_join = pd.merge(plot_data, analomaly_dates, how='left', on='week_number')
    plot_data_join['status_text'].fillna('normal', inplace=True)
    plot_data_join.set_index('datetime', inplace=True)

    fig = px.line(plot_data_join, x=plot_data_join.index , y=[target_col], color='status_text', color_discrete_sequence=['#00b32c', '#e03531'], title=component)
    analomaly_dates.rename(columns={'status_text':'Health_Status'}, inplace=True)
    display_anomaly_data = (
    analomaly_dates
    .filter(['Start_Date','End_Date','Health_Status'])
    .reset_index(drop=True)
    )

    anomaly_count = len(analomaly_dates)
    anomaly_pct ="{0:.0%}".format(len(analomaly_dates)/len(eval_df))
    metric_data = [anomaly_count, anomaly_pct]
    

    analomaly_dates['component'] = component

    return analomaly_dates, fig, metric_data



# "EDP/20230424_t07_Gen_Bear2_Temp_Avg_v1_rf.pickle"
# target_col = 'Gen_Bear2_Temp_Avg'
# features2rem= [
#     'Turbine_ID',
#     'HVTrafo_Phase1_Temp_Avg',
#     'HVTrafo_Phase2_Temp_Avg',
#     'HVTrafo_Phase3_Temp_Avg',
#     'Gen_Phase1_Temp_Avg',
#     'Gen_Phase2_Temp_Avg',
#     'Gen_Phase3_Temp_Avg',
#     'Gen_Bear_Temp_Avg',
#     'Gen_SlipRing_Temp_Avg'
#     ]
#     ocsvm_model_path = 'EDP/weekly_anomaly_detector_rf.pickle'