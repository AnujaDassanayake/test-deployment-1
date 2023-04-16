import pickle
import pandas as pd

def anomaly_detection_daily(model_path, test_metrics):
    
    pick_read = open(model_path,'rb')
    anomaly_detector_model = pickle.load(pick_read)
    pick_read.close()

    all_metrices = test_metrics[['MAE','MSE','RMSE','MAPE','R2']]
    test_metrics_copy_all = test_metrics.copy()
    test_metrics_copy_all['status'] = anomaly_detector_model.predict(all_metrices)
    test_metrics_copy_all['status_text'] = test_metrics_copy_all['status'].replace({-1: 'anomaly', 1: 'normal'})

    test_metrics_copy_all['date'] = test_metrics_copy_all['Time_Period'].str[:10]
    order = ['Time_Period','date','rows','MAE','MSE' ,'RMSE' ,'MAPE' ,'R2' ,'completeness' ,'status' ,'status_text']
    test_metrics_copy_all = test_metrics_copy_all.reindex(columns=order)

    return test_metrics_copy_all
