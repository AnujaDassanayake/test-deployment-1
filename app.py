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



from evaluation import (
    data_for_time_window_mlr,
    evaluate_model,
    calculate_error_matrices
)

from anomaly_detection import (
  anomaly_detection
)

st.set_page_config(layout='wide')
st.write("""
    # Wind Turbine Anomaly Detector
    """)
st.sidebar.header('Upload SCADA Data as CSV')
uploaded_file  = st.sidebar.file_uploader(
        label = "Upload SCADA file for testing",
        type= ['csv']
        )

tab1, tab2, tab3= st.tabs(["Stats", "Anomaly Detector", "Variables"])

with tab1:
    
    # st.write("""
    # # Enter SCADA data
    # """)

    # uploaded_file  = st.file_uploader(
    #     label = "Upload SCADA file for testing",
    #     type= ['csv']
    #     )
    
    uploaded_dataframe = None

    if uploaded_file is not None:
        
        #reading the uploaded CSV
        uploaded_dataframe = pd.read_csv(uploaded_file, sep=';')
        uploaded_dataframe['Timestamp'] = pd.to_datetime(uploaded_dataframe['Timestamp'])
        uploaded_dataframe_anomaly = uploaded_dataframe.copy()
        uploaded_dataframe['Date'] = uploaded_dataframe['Timestamp'].dt.date


        #Calculating the number of records present in the dateset
        num_row_dataset = "{:,}".format(len(uploaded_dataframe))

        #Displaying the number of records
        # st.write("### Number of records = " + num_row_dataset)

        #Capturing the unique wind turbines present in the dataset
        available_turbines = uploaded_dataframe['Turbine_ID'].unique()

        #Creating a list of componets/ fields present in the dataset
        components = uploaded_dataframe.columns.to_list()[2:]

        #Calculating the number of records available for each wind turbine
        #Values in Gen_RPM_Max is taken as the proxy for counting the records
        records = (
                    uploaded_dataframe[['Turbine_ID','Gen_RPM_Max']]
                    .groupby('Turbine_ID').count()
                    )
        
        #Creating a new column named 'Available Record Count'
        records['Available Record Count'] = records['Gen_RPM_Max']
        
        #Ploting the pie chart for the available records per turbine
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=records.index, values=records['Available Record Count'], name='Population of European continent'),
            1, 1)
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")
        # st.plotly_chart(fig, use_container_width=True)
        #The pie chart was not included as it didn't convey any insight

        #Extracting Year and Month from the original timestamp
        uploaded_dataframe['Year'] = uploaded_dataframe['Timestamp'].dt.year
        uploaded_dataframe['Month'] = uploaded_dataframe['Timestamp'].dt.month

        #Calculating the average power production for each year-month present in the dataset
        monthly_production_farm = (
            uploaded_dataframe
            .groupby(['Year','Month'], as_index=False)
            .agg(
                Average_production = ('Grd_Prod_Pwr_Avg','mean')
            )       
        )      

        #Ploting the average monthly production in the full wind farm as a bar chart
        fig = go.Figure(
            go.Bar(
            name = 'Power Production', 
            x = monthly_production_farm['Month'], 
            y = monthly_production_farm['Average_production']
            ),
        )

        fig.update_layout(
                title = dict(text = 'Average Power Production of all turbines'),
                xaxis_title="Month",
                yaxis_title="Avg Power Produced (KWh)"
            )  
        
        fig.update_traces(
                        marker = dict(
                        color = monthly_production_farm['Average_production'],
                        colorscale = 'aggrnyl'
                        )
        )          
        
        st.plotly_chart(fig, use_container_width=True)

        #Ploting the average monthly production of each turbine in the farm
        fig = go.Figure()   

        for turbine in available_turbines:
            uploaded_dataframe_filtered = uploaded_dataframe.loc[uploaded_dataframe['Turbine_ID'] == turbine]
            monthly_production_select = (
                uploaded_dataframe_filtered
                .groupby(['Year','Month'], as_index=False)
                .agg(
                    Average_production = ('Grd_Prod_Pwr_Avg','mean')
                )   
            )
            fig.add_trace(
                go.Bar(
                x = monthly_production_select['Month'], 
                y = monthly_production_select['Average_production'],
                name = turbine,
                marker = dict(
                            color = monthly_production_farm['Average_production'],
                            colorscale = 'aggrnyl'
                            )
                )

            )
            fig.update_layout(
                    title = dict(text = 'Average Power Production'),
                    xaxis_title = 'Months',
                    yaxis_title="Avg Power Produced (KWh)",
                    barmode ='group'
                )        
        st.plotly_chart(fig, use_container_width=True)

        
        st.write('### Windrose plot for the wind farm')

        #creating 2 columns for the windrose graph
        windrose_col1, windrose_col2 = st.columns(2)
        
        with windrose_col2:
            #plotting the windrose graph
            
            #Defining the min and max dates for the date range slider for drawing the windrose
            min_date_windrose = uploaded_dataframe['Date'].min()
            max_date_windrose = uploaded_dataframe['Date'].max()

            #Adding a date range slider to get start date and end date for the drawing the windrose
            windrose_range = st.slider(
                "Select date range for the windrose plot:",
                value=(min_date_windrose, max_date_windrose),
                min_value = min_date_windrose,
                max_value = max_date_windrose
                )
            
            st.write("Displaying windrose from ", windrose_range[0]," to ", windrose_range[1])

        with windrose_col1:

            #Selecting the start date and end date from the windrose_range tuple
            windrose_sdate = windrose_range[0]
            windrose_edate = windrose_range[1]

            #Filtering date range for drawing windrose        
            data_for_windrose = uploaded_dataframe[(uploaded_dataframe['Date'] > windrose_sdate ) & (uploaded_dataframe['Date'] < windrose_edate)]

            ax = WindroseAxes.from_ax()
            ax.bar(data_for_windrose['Amb_WindDir_Abs_Avg'], data_for_windrose['Amb_WindSpeed_Avg'], normed=True, opening=0.8, edgecolor='white')
            ax.set_legend()
            plt.savefig('windrose.jpg')
            st.image('windrose.jpg')

        
        st.write("### Start-End dates")
        
        uploaded_dataframe['Dates'] = uploaded_dataframe['Timestamp'].dt.date

        start_end = (
            uploaded_dataframe
        .groupby('Turbine_ID')
        .agg(
            Start_Date = ('Timestamp', 'min'), 
            End_Date = ('Timestamp', 'max'),
            Number_of_Days = ('Dates', 'nunique')
            )
            )

        start_end['Start_Date'] = start_end['Start_Date'].dt.date
        start_end['End_Date'] = start_end['End_Date'].dt.date
        st.write(start_end) 

        uploaded_dataframe
        
    else:
        st.warning("""
        # Please Enter SCADA data to proceed!
        """)
        
with tab2:

    st.write("""
    # Identified Anomalies
    
    """)
   
    if uploaded_dataframe is not None:
        with st.spinner('Anomaly detection model pipeline executing'):
            time.sleep(5)
            turbine = st.selectbox(
                'Select Turbine ID',
                available_turbines
                )

            component = st.selectbox(
                'Select component',
                components
                )
            if (turbine == "T07") & (component == "Gen_Bear2_Temp_Avg"):
                #Chaning data type of the timestamp column
                uploaded_dataframe_anomaly['Timestamp'] = pd.to_datetime(uploaded_dataframe_anomaly['Timestamp'])
                uploaded_dataframe_anomaly.set_index('Timestamp',inplace = True)
                selected_turbine_scada_data = uploaded_dataframe_anomaly[uploaded_dataframe_anomaly['Turbine_ID'] == turbine]
                selected_turbine_scada_data = selected_turbine_scada_data.sort_values(by = ['Timestamp'], ascending = True)
                selected_turbine_scada_data.dropna(inplace=True)
                #len(selected_turbine_scada_data)

                dataset = selected_turbine_scada_data

                pick_read = open("EDP/20230322_t07_Gen_Bear2_Temp_Avg_v1.pickle",'rb')
                model1 = pickle.load(pick_read)
                pick_read.close()

                # target_col = 'Gen_Bear2_Temp_Avg'
                target_col = component

                features2rem= [
                    'Turbine_ID',
                    'HVTrafo_Phase1_Temp_Avg',
                    'HVTrafo_Phase2_Temp_Avg',
                    'HVTrafo_Phase3_Temp_Avg',
                    'Gen_Phase1_Temp_Avg',
                    'Gen_Phase2_Temp_Avg',
                    'Gen_Phase3_Temp_Avg',
                    'Gen_Bear_Temp_Avg',
                    'Gen_SlipRing_Temp_Avg'
                    ]
                    
                eval_df = calculate_error_matrices(dataset ,model1 ,target_col, features2rem)
                eval_df

                model_path = 'EDP/daily_anomaly_detector.pickle'
                test_metrics_copy_all = anomaly_detection(model_path, eval_df)
                # st.line_chart(dataset.Amb_WindSpeed_Est_Avg)
                test_metrics_copy_all['date'] = pd.to_datetime(test_metrics_copy_all['date'])
                test_metrics_copy_all.set_index('date',inplace = True)

                analomaly_dates = (
                test_metrics_copy_all
                .query('status_text == "anomaly"')
                )

                data_for_cal_plot = analomaly_dates[['status_text']]
                data_for_cal_plot
                # calplot.yearplot(data_for_cal_plot['status'])
            else:
                st.warning("""
                Anomaly detector model is unavailable for the selected turbine + componenet combination 
                """)
        st.success('Done!')


# with tab3:

#     st.write("""
#     # Wind Turbine Stats
#     """)
#     if uploaded_dataframe is not None:    

#         turbine_tab3 = st.selectbox(
#             ' Select Turbine ID',
#                 available_turbines
#                 )

        # SCADA_data_tab3 = uploaded_dataframe[['Turbine_ID'] == turbine_tab3]

        # monthly_power = uploaded_dataframe
    
with tab3:
    st.write("""
    # Variable description
    """)
    
    varible_desc_path = 'EDP/variables.csv'
    varible_desc = pd.read_csv(varible_desc_path)

    st.write(varible_desc)




css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
