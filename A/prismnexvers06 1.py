import streamlit as st
import io
import requests
import base64  # เพิ่มการนำเข้าโมดูล base64
import math
import json
import time
import datetime
import numpy as np
import pandas as pd 
import ipywidgets as widgets
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from IPython.display import display
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from datetime import time, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from bokeh.plotting import figure, show
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from math import sqrt
from plotly.subplots import make_subplots
from prophet import Prophet
#===============================================================================================================================
# -------------------------------------------------------------------------------------------------------------------
def home():
    
    if 'token' not in st.session_state:
        st.session_state.token = None
    # if 'asset' not in st.session_state:
    #     st.session_state.asset = None

    if 'asset_id' not in st.session_state:
        st.session_state.asset_id = None

    if 'df_point' not in st.session_state:
        st.session_state.df_point = None

    if 'project_id' not in st.session_state:
        st.session_state.project_id  = None
    

    asset_id = st.session_state.asset_id

    with st.sidebar:
        selected2 = option_menu("Menu",  ["Login/Logout","Homepage","analysis", "overallhealth","forecasting"], 
            icons=['box-arrow-in-right','house', 'graph-up', 'activity','graph-up-arrow'], menu_icon="cast",  default_index=1)
        st.sidebar.success(selected2)
#--------------------------------------------------------------------------------------------------------------------------

    if selected2 == "Login/Logout":
        
        token = login()
        # print(token)
        # print(token) //pass
        st.session_state.token = token
#--------------------------------------------------------------------------------------------------------------------------

    elif selected2 == "Homepage":
        st.header("Welcome to Homepage", divider='grey')
        token = st.session_state.token 
        if token != []:
            asset_id = asset(token)
            # print(asset_id)
            asset_id = st.session_state.asset_id
            if asset_id is not None:
                project_id = project(asset_id,token)
                st.session_state.project_id  = project_id 
                df_point = receivepoint(project_id,token)
                st.session_state.df_point = df_point
                print(st.session_state.df_point)
                print(project_id)
        else:
            st.warning("Please Login Again")
#--------------------------------------------------------------------------------------------------------------------------
    elif selected2 == "analysis":
        st.header("Welcome to analysis data", divider='grey')

        if 'df' not in st.session_state:
            st.session_state.df = None
        # print(asset_id)
        if asset_id is not None:
            # st.header("Welcome to plot graph", divider='grey')
            selected = option_menu("",  ["created dataset","plotgraph"], 
                icons=['clock-history', 'graph-up',],  default_index=0,orientation="horizontal")    
            
            if selected == "created dataset":
                st.caption('You should choose two time periods to create a data set.')
                if st.session_state.project_id is not None:
                    project_id = st.session_state.project_id
                    token = st.session_state.token 
                    df_point = st.session_state.df_point
                    selected_values1 = st.multiselect('select point_id:', df_point['names'])
                    
                    point_ids  = df_point[df_point['names'].isin(selected_values1)]
                    selected_values = df_point[df_point['names'].isin(selected_values1)]
                    selected_values = selected_values['names1'].tolist()

                    point_ids = point_ids['point_ids'].tolist()
                    
                    name = pd.DataFrame({'selected_values': selected_values, 'selected_values1': selected_values1})

                    dataactual = timeperiod1(point_ids,token)

                    print(dataactual)
                    typedatatraining = st.radio(
                                    "Type Data Training",
                                    
                                    ["upload_CSV files", "API datatraing"])
                    if typedatatraining == "API datatraing":
                        df_training = datatraining_id(project_id,token)
                        
                        selected_datatraining = st.selectbox('select datatrining:', df_training['trainingDataSetName'])
                        selected_datatraining_id  = df_training.loc[df_training['trainingDataSetName'] == selected_datatraining, 'trainingDataSetId'].values.tolist()
                        selected_datatraining_id= selected_datatraining_id[0]
                        datatraining = datatrainingdata(selected_datatraining_id,selected_values,token)
                        if datatraining != []:
                            datatraining =  datatraining[selected_values]
                            datatraining = datatraining.assign(Class=selected_datatraining)

                    elif typedatatraining == "upload_CSV files":
                        uploaded_file = st.file_uploader("Choose a CSV file")
                        if uploaded_file is not None:
                            bytes_data = uploaded_file.read()
                            st.write("filename:", uploaded_file.name)
                            try:
                                if not bytes_data:  # ตรวจสอบว่ามีข้อมูลในไฟล์หรือไม่
                                    st.write("The uploaded file is empty.")
                                else:
                                    # ใช้ io.StringIO เพื่ออ่านข้อมูล CSV
                                    data = io.StringIO(bytes_data.decode('utf-8'))
                                    df = pd.read_csv(data)
                                    df = df.reset_index(drop=True)
                                    df = df.set_index(df.columns[0])
                                    df = df.rename_axis("timeStamp")
                                    nametrain = uploaded_file.name
                                    print(df)
                                    # df.columns = [f'{col}({df.iloc[2][col]})' for col in df.columns]
                                    # df = df[4:]
                                    # print(df)
                                    # ลบแถวที่มีค่า NaN
                                    # df = df.dropna()
                                    # # ลบคอลัมน์ที่มีค่า NaN
                                    # df = df.dropna(axis=1)  
                                    # ตรวจสอบว่ามีคอลัมน์ที่ตรงกับ selected_values1 หรือไม่
                                    
                                    matching_columns = [col for col in selected_values1 if col in df.columns]

                                    print('selected_values1:',selected_values1)

                                    print('matching_columns:',matching_columns)
                                    if len(selected_values1) >0:
                                        # เลือกเฉพาะคอลัมน์ที่ตรงกับ selected_values1
                                        datatraining = df[selected_values]
                                        datatraining = datatraining[4:]
                                        datatraining.rename(columns=dict(zip(selected_values, selected_values1)), inplace=True)
                                        datatraining.index = pd.to_datetime(datatraining.index)

                                        
                                        datatraining = datatraining.assign(Class=nametrain)
                                         
                                    else:
                                        st.error("Please upload a new file with matching column names.")
                                        
                            except Exception as e:
                                st.write("An error occurred:", e)

                    

                    if st.button("submit"):
                        dataactual = dataactual.assign(Class='Actual')

                        if typedatatraining == "API datatraing":
                            df = [dataactual,datatraining]
                            column_rename_mapping = {}
                            for idx, row in name.iterrows():
                                column_rename_mapping[row['selected_values']] = row['selected_values1']
                            for i in range(len(df)):
                                df[i].rename(columns=column_rename_mapping, inplace=True)
                            
                            st.success("Please select the plot graph page")
                            st.caption("historical data:")
                            st.dataframe(dataactual) 
                            st.caption("data training:")
                            st.dataframe(datatraining) 

                        elif typedatatraining == "upload_CSV files":
                            column_rename_mapping = {}

                            for idx, row in name.iterrows():
                                column_rename_mapping[row['selected_values']] = row['selected_values1']

                            dataactual = dataactual.rename(columns=column_rename_mapping)
                            df = [dataactual,datatraining]
                            
                            st.success("Please select the plot graph page")
                            st.caption("historical data:")
                            st.dataframe(dataactual) 
                            st.caption("data training:")
                            st.dataframe(datatraining) 
                        
                        st.session_state.df = df 
                        return df
                        
            if selected == "plotgraph":
                if st.session_state.df is not None:
                    df= st.session_state.df
                    print(df)
                    plot_graph(df)
                else:
                    st.error("Please created dataset")
        else:
            st.warning("Please select asset")

    elif selected2 == "overallhealth":
        st.header("Welcome to overallhealth", divider='grey')
        if asset_id is not None:
            if st.session_state.df_point is not None:
                    df_point = st.session_state.df_point
                    token = st.session_state.token
                    df = timeforOMR(df_point,token)
                    if df is not None:
                        
                        name = df[0]
                        df = df[1]
                        if df.empty :
                            st.error("Empty DataFrame")
                        else:
                            print('DF:',df)
                            df['residual or prediction error'] = abs(df['actual'] - df['predict'])
                            print(df)

                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

                            # Plot the first trace in the first row
                            for column in df.columns[:-1]:  # Exclude the last column for subplot
                                fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column), row=1, col=1)

                            # Plot the subplot in the second row
                            column_for_subplot = df.columns[2]  # Change this to the desired column
                            fig.add_trace(go.Scatter(x=df.index, y=df[column_for_subplot], mode='lines', name=column_for_subplot), row=2, col=1)

                            # Update layout for the entire figure
                            fig.update_layout(
                                title='Overall Health',
                                
                                showlegend=True,
                                xaxis=dict(tickangle=-45),
                            )

                            # Update layout for the legend
                            fig.update_layout(legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ))

                            # Show the figure using Streamlit
                            st.plotly_chart(fig, theme=None)
        else:
            st.warning("Please select asset") 
        
#--------------------------------------------------------------------------------------------------------------------------

    elif selected2 == "forecasting":
        st.header("Welcome to forecasting", divider='grey')
        if asset_id is not None:
            # st.header("Welcome to plot graph", divider='grey')
            if st.session_state.df_point is not None:
                type = ["linear","quadratic","Prophet","LSTM"]
                typetoforecast = st.selectbox("please select", type)
                df_point = st.session_state.df_point
                token = st.session_state.token 
                df1 = timespanforecast(df_point,token)
                # print(df[2])
                
                # print(df)
                if df1 is not None:
                    name = df1[0]
                    df = df1[1]
                    if df.empty :
                        st.error("Empty DataFrame")
                    else:
                        day = df1[2]
                        frequen = df1[3]
                        if st.button("timeseries"):

                            fig = go.Figure()

                            # Add a line plot for the 'SCG.PP3IP21.II0401P' column
                            fig.add_trace(go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1], mode='lines'))

                            fig.update_layout(
                                title='Time Series Plot',
                                xaxis_title='Timestamp',
                                yaxis_title=name,
                            )
                            fig.update_layout(legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ))
                            st.plotly_chart(fig, theme=None)
                            st.warning("Are you sure")
                            
                        if st.button("plot forecasting"):
                            if typetoforecast == "linear":
                                linear(name,df,day,frequen)
                            if typetoforecast == "quadratic":
                                quadractic(name,df,day,frequen)
                            if typetoforecast == "LSTM":
                                neural(name,df,day,frequen)
                            if typetoforecast == "Prophet":
                                prophetforecast(name,df,day,frequen)
                        else:
                            st.warning("Please observe the trend with the time series graph.")        
                else:
                    st.warning("Please create df")
        else:
            st.warning("Please select asset")
#--------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def datatrainingdata(selected_datatraining_id,selected_values,token):
    print(selected_values)
    
    # name = pd.Datafrasme[selected_values,selected_values1]
    url = "https://repcrydrpws71/aveva.pa.webapi/api/Project/training-dataset?dataSetId={}".format(selected_datatraining_id)
    payload = {}
    headers = {
        'Authorization': 'Bearer ' + token
    }
    response = requests.request("GET", url, headers=headers, data=payload,verify=False)
    response = response.text
    data = json.loads(response)
    # print(data)
    if "ErrorDetails" in data:
            st.error(data["ErrorDetails"])
    else:
        data_dict = {
            'timeStamp': [],
        }

        # Process the data and populate the dictionary
        for entry in data['trainingData']:
            data_dict['timeStamp'].append(entry['timeStamp'])
            sensor_info = entry['sensorInformation']
            for sensor_data in sensor_info:
                data_dict[sensor_data['sensorName']] = data_dict.get(sensor_data['sensorName'], [])
                data_dict[sensor_data['`sensorName`']].append(sensor_data['value'])
        
        # Create the DataFrame
        df = pd.DataFrame(data_dict)

        # Set the timestamp as the index
        df.set_index('timeStamp', inplace=True)
        # print(df)
        # Display the DataFrame
        
        
        # print(df)
        return df 
# -------------------------------------------------------------------------------------------------------------------
def timeperiod1(point_ids,token):
    # st.checkbox(df_point)
    current_datetime = datetime.datetime.now()
    # print(type)
    # ลบ 3 ปีจากวันที่และเวลาปัจจุบัน
    start = current_datetime - datetime.timedelta(days=365*2)
    # ลบ 12 ชั่วโมงจากวันที่และเวลาปัจจุบัน

    end = current_datetime - datetime.timedelta(days=1)
    # print(start,end)
    st.caption('actual :')

    col1, col2,col3 = st.columns(3)

    with col1:

        start_date1 = st.date_input("select start date1", end.date(),start.date(),end.date())
        start_time1 = st.time_input("select start time1", datetime.time(0, 0))
        
        
    with col2:
        # เลือกเวลาสิ้นสุด
        end_date1 = st.date_input("select end date1", end.date(),start.date(),end.date()) 
        end_time1 = st.time_input("select end time1", datetime.time(0, 0))

    with col3:
        frequency = st.radio(
            "frequecy actual data",
            ["5 minute", "10 minute", "15 minute", "20 minute", "25 minute"],
            index=None,
        )
    
    start_datetime1 = pd.Timestamp(datetime.datetime.combine(start_date1, start_time1))
    end_datetime1 = pd.Timestamp(datetime.datetime.combine(end_date1, end_time1))
    

    start_datetime1 = start_datetime1.to_pydatetime()
    end_datetime1 = end_datetime1.to_pydatetime()
    
    if frequency is not None:
        if frequency == "5 minute":
            frequency = 5
        elif frequency == "10 minute":
            frequency = 10
        elif frequency == "15 minute":
            frequency = 15
        elif frequency == "20 minute":
            frequency = 20
        elif frequency == "25 minute":
            frequency = 25
    
        period1 = [start_datetime1, end_datetime1,frequency]
        
        data1 = histarical1(point_ids,period1,token)    

        if data1 is not None:
            
            df_list1 = []

            for item in data1:
                point_name = item["PointName"]

                for data_point in item["DataPoints"]:
                    timestamp = data_point["TimeStamp"]
                    value = data_point["Value"]
                    df_list1.append({"PointName": point_name, "TimeStamp": timestamp, "Value": value})

            # สร้าง DataFrame จากรายการข้อมูล
            df = pd.DataFrame(df_list1)
            # print(df)

            df = df.pivot(index='TimeStamp', columns='PointName', values='Value')
            name = df.columns[0]
            value = df.iloc[:, 0]
            
            # print(df1)

            return df
                
        else: 
            st.warning("please selected timespan")

    

    
# -------- -------------------------------------------------------------------------------------------------------
def histarical1(point_ids,period1,token) :

    point_ids = np.array(point_ids)
    point_ids = point_ids.tolist()
    print(point_ids)
    if period1 is not None:

        starttime = period1[0]
        starttime = starttime.strftime("%m/%Y/%d %H:%M:%S")

        endtime = period1[1]
        endtime = endtime.strftime("%m/%Y/%d %H:%M:%S")
        frequency = period1[2]
        #-------------------------------------------------------------------------
        url = "https://repcrydrpws71/api/v1/histdata/"

        # สร้าง payload โดยแทนค่า "PointIDs" ด้วย project_ids
        payload = json.dumps({"PointIDs": point_ids,"StartDateTime": starttime,"EndDateTime":endtime,"FrequencySeconds":60*frequency})

        headers = {
        'Content-Type': 'application/json','Authorization': 'Bearer '+token }

        response = requests.request("POST", url, headers=headers, data=payload,verify=False)
        response = response.text
        response = json.loads(response)
        if "ErrorDetails" in response:
            st.error(response["ErrorDetails"])
        else:
            return response
        
#-----------------------------------------------------------------------------------------------------------------------------------
def datatraining_id(project_id,token):
    url = "https://repcrydrpws71/aveva.pa.webapi/api/Project?projectId={}".format(project_id)
    payload = {}
    headers = {
        'Authorization': 'Bearer ' + token
    }
    response = requests.request("GET", url, headers=headers, data=payload,verify=False)
    response = response.text
    data = json.loads(response)
    # เข้าถึงค่า "UserGuid" และเก็บในตัวแปร userid
    df = data["dataSetInfo"]
    trainingDataSetId = []
    trainingDataSetName = []

    # ใช้ลูป for เพื่อดึงค่า ProjectId และเก็บใน project_ids
    for item in df:
        # สร้าง array ใหม่เพื่อเก็บค่า ProjectId
        trainingDataSetId.append(item['trainingDataSetId'])
        trainingDataSetName.append(item['trainingDataSetName'])
    # สร้าง DataFrame
    df_train = pd.DataFrame({'trainingDataSetId': trainingDataSetId, 'trainingDataSetName': trainingDataSetName})
    # print(df_train)
    return df_train

#===================================================================================================================
def receivepoint(project_id,token):           
    url = "https://repcrydrpws71/api/v1/projects/{}".format(project_id)
    metrix = metixname(project_id,token)
    payload = {}
    headers = {
    'Authorization': 'Bearer '+token
    }

    response = requests.request("GET", url, headers=headers, data=payload,verify=False)
    
    # แปลง JSON string เป็น Python dictionary
    data = json.loads(response.text)

    # เข้าถึงค่า "UserGuid" และเก็บในตัวแปร userid
    df = data["Points"]
    point_ids = []
    name = []

    # ใช้ลูป for เพื่อดึงค่า ProjectId และเก็บใน project_ids
    for item in df:
        # สร้าง array ใหม่เพื่อเก็บค่า ProjectId
        point_ids.append(item['ProjectPointId'])
        name.append(item['Name'])
    # สร้าง DataFrame
    df_point1 = pd.DataFrame({'point_ids': point_ids, 'names': name})
    df_point1['Description'] = df_point1.apply(lambda row: row['names'] + '(' + metrix[metrix['PointName'] == row['names']]['MetricPointName'].values[0] + ')', axis=1)
    df = df_point1 
    df = df.rename(columns={'names': 'names1'})
    df = df.rename(columns={'Description': 'names'})
    return df      

def metixname(project_id,token) :
    url = "https://repcrydrpws71/api/web/projectprofiles"

    payload = json.dumps({
    "ProjectIds": [
        project_id  
    ],
    "OnlyPointsInAlarm": False,
    "IncludeForecastInformation": True
    })
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer '+token
    }

    response = requests.request("POST", url, headers=headers, data=payload,verify=False) 
    data = json.loads(response.text)
    PointName = []
    MetricPointName = []
    for item in data:
        if item.get('TypeName' ) == 'Point':
            PointName.append(item['PointName'])
            MetricPointName.append(item.get('MetricPointName')) 
    data = {'PointName': PointName, 'MetricPointName': MetricPointName}
    metrix = pd.DataFrame(data)    
    return metrix

# -------------------------------------------------------------------------------------------------------------------
def login():
    token = []
    with st.form("my_form"):
        st.image('settings.png',  width=150)
        col1, col2 = st.columns(2)
        with col1:
            beginusername = st.text_input(
                "Username ",
                key="unique_key1"
            )

            st.caption("EX.CEMENTHAI\\user")
        with col2:
            beginpassword = st.text_input(
                "Password ",
                type="password",
                key="unique_key2"
            )
        Logined = st.form_submit_button("Login")

    if Logined:   
        username = st.session_state.username = beginusername
        password = st.session_state.password = beginpassword
        userid = None
        url = "https://repcrydrpws71/api/v1/identity"
        # สร้างคู่คีย์ (username, password) สำหรับ authentication
        payload = {}
        auth_header = 'Basic ' + base64.b64encode(f'{username}:{password}'.encode()).decode()
        # สร้าง headers
        headers = {
            'Authorization': auth_header
        }
        responseuser = requests.request("POST", url, headers=headers, data=payload,verify=False)
        responseuser_text = responseuser.text
        response = json.loads(responseuser_text)
        # เช็คว่ามี "ErrorDetails" ใน response หรือไม่
        if "ErrorDetails" in response:
            st.error(response["ErrorDetails"])
        else:
            userid = response["UserGuid"]
            url = "https://repcrydrpws71/token"
            payload = 'grant_type=password&username='+userid+'&client_id=prism-api&client_secret=prism-api'
            headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
            }
            responsetoken = requests.request("POST", url, headers=headers, data=payload,verify=False)
            # JSON string จาก response.text
            responsetoken = responsetoken.text
            # แปลง JSON string เป็น Python dictionary
            datatoken = json.loads(responsetoken)
            expires_str = datatoken[".expires"]
            expires_datetime = datetime.datetime.strptime(expires_str, "%a, %d %b %Y %H:%M:%S GMT")
            # ตรวจสอบว่า Token หมดอายุหรือไม่โดยเปรียบเทียบกับวันปัจจุบัน
            current_datetime = datetime.datetime.utcnow()

            if expires_datetime <= current_datetime:
                st.warning('Please enter your username and password again')
                # st.experimental_rerun()
            else:
                st.success('login success')
                token = datatoken["access_token"]
                st.warning('Please select home page')
    else:
        st.warning('Please enter your username and password')
    return token 
#---------------------------------------------------------------------------------------------------------------------------
def asset(token):
        # ตั้งค่าเริ่มต้นของ asset_id
    if token is not None:
        if 'asset_id' not in st.session_state:
            st.session_state.asset_id = 1

        if 'df' not in st.session_state:
            st.session_state.asset_id = None

        # ลูป while ในการดึงข้อมูล Asset จาก API จนกว่า ChildAssets จะเป็น None
        while st.session_state.asset_id != 0:
            asset_id = st.session_state.asset_id
            url = "https://repcrydrpws71/api/v1/assets/{}".format(asset_id)
            payload = {}
            # token = str(token)
            headers = {
                'Authorization': 'Bearer ' + token
            }
            response = requests.get(url, headers=headers, data=payload, verify=False)
            response = response.text
            
            data = json.loads(response)
            if "ErrorDetails" in data:
                asset_id = None
                st.session_state.asset_id  = asset_id
                st.error(response["ErrorDetails"])
            else:
                if len(data['ChildAssets']) >= 1 :
                    df = data['ChildAssets']

                    AssetId = []
                    Description = []

                    # ใช้ลูป for เพื่อดึงค่า AssetId และ Description และเก็บในรายการ
                    for item in df:
                        AssetId.append(item['AssetId'])
                        Description.append(item['Description'])
                    
                    data1 = [AssetId, Description]
                    df = pd.DataFrame(data1)
                    df = df.transpose()
                
                    selected_value = st.selectbox('selected asset:', df[1])
                    asset_id = df[df[1] == selected_value][0].values[0]
                    
                    # อัปเดตค่า asset_id ใน session state เพื่อใช้ในการร้องขอข้อมูลต่อไป
                    st.session_state.asset_id = asset_id
                    
                elif len(data['ChildAssets']) == 0 and len(data['ChildProjects']) >= 1 :
                    break

                # elif len(data['ChildProjects']) >= 1 :
                #     break
                elif len(data['ChildAssets']) == 0 and len(data['ChildProjects']) == 0 :
                    asset_id = None
                    st.session_state.asset_id = asset_id
                    st.error("Please select new Asset")
                    break

        return asset_id    
        
    else: 
        st.warning("Please Login")
            
# -------------------------------------------------------------------------------------------------------------------
def project(asset_id,token):

    url = "https://repcrydrpws71/api/v1/assets/{}".format(asset_id)

    payload = {}
    headers = {
    'Authorization': 'Bearer '+token
    }

    response = requests.request("GET", url, headers=headers, data=payload,verify=False)
    response =response.text
    data = json.loads(response)
    if "ErrorDetails" in data:
        da = None
        project_id = None
        st.error(response["ErrorDetails"])
    else:
        df = data["ChildProjects"]

        project_ids = []
        name = []

        # ใช้ลูป for เพื่อดึงค่า ProjectId และเก็บใน project_ids
        for item in df:
            project_ids.append(item['ProjectId'])
            name.append(item['Name'])

        df_point = [project_ids,name]
        df = pd.DataFrame(df_point)
        df = df.transpose()

        if not df.empty :
            
            selected_value = st.selectbox('selected project:', df[1])
            project_id = df[df[1] == selected_value][0].values[0]

            if st.button("Submit"):
                if project_id is not None:
                    
                    st.success("please select analysis ,overall health and forecasting page ")
                    
            else: 
                st.warning("please submit")
        else: 
            st.warning("Please select the timespan page to select the data range.")
    return project_id
#----------------------------------------------------------------------------------------------------------------
def plot_graph(df):
    if 'x_array' not in st.session_state:
        st.session_state.x_array = []

    if 'data_choices' not in st.session_state:
        st.session_state.data_choices = []

    typetoplot = st.sidebar.selectbox("Select type:", ["select types","static plot", "specialty"])
    if typetoplot == "select types":
        selecttypes_page()
    elif typetoplot == "static plot":
        subplot = st.sidebar.selectbox("select plot : ",["select plot","single plot","subplot"])
        if subplot == "select plot":
            st.warning("Please select Plot")
        elif subplot == "single plot":

            typegraph = st.selectbox("Select graph:", ["select graph","timeseries", "distribution", "scatter", "boxplot","violine"])
            if typegraph  == "timeseries":
                dftime=timeseries(df)    

                columns_to_plot = dftime.columns[0:]
                for column in columns_to_plot:
                    fig = px.line(dftime, x=dftime.index, y=dftime.columns)
                    fig.update_layout(
                        # title='Timeseries',
                        xaxis_title='TimeStamp',
                        yaxis_title='Value'
                    )

                if st.button("Plot"):
                    fig.update_layout(
                        legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ))
                    st.plotly_chart(fig, theme=None)

            elif typegraph  == "distribution":
                dx_values=distribution_page(df)

                print(dx_values)
                fig = px.histogram(dx_values, x=dx_values.columns[0], color="Class", nbins=8)
                fig.update_layout(
                    xaxis_title=dx_values.columns[0],
                    yaxis_title='Frequency',
                    title='Histogram'
                )

                if st.button("Plot"):
                    fig.update_layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ))

                    st.plotly_chart(fig, theme=None)
                
            elif typegraph == "scatter":
                scatter_result=scatter_page(df)
                # print(scatter_result)
                
                x_values = scatter_result.iloc[:, 0]
                y_values = scatter_result.iloc[:, 1]
                x_name = scatter_result.columns[0]
                y_name = scatter_result.columns[1]
                # กำหนดชุดสีที่คั่นต่างกันสำหรับแต่ละคลาส
                color_sequence = ['#3366FF','#FF5733']  # ตัวอย่างสีที่แตกต่างกัน

                fig = px.scatter(scatter_result, x=x_values, y=y_values, color='Class', title='Scatter Plot',
                                color_discrete_sequence=color_sequence)
                
                fig.update_xaxes(title_text=x_name)
                fig.update_yaxes(title_text=y_name)
                if st.button("Plot"):
                    fig.update_layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ))
                    st.plotly_chart(fig, theme=None)

            elif typegraph == "boxplot":
                df_boxplot=boxplot_page(df)
                # print(df_boxplot)     
                if df_boxplot is not None:
                    fig = px.box(df_boxplot, title='Box Plot', color='Class')

                    if st.button("Plot"):
                        fig.update_layout(legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ))
                        st.plotly_chart(fig, theme=None)

            elif typegraph == "violine":
                df_violine=violine_page(df) 
                if df_violine is not None:
                    fig = px.violin(df_violine, title='Violin Plot', color='Class')

                    if st.button("Plot"):
                        fig.update_layout(legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ))
                        st.plotly_chart(fig, theme=None)

        elif subplot == "subplot":

            typegraph = st.selectbox("Select graph:", ["select graph","timeseries", "distribution", "scatter", "boxplot","violine"])

            if  typegraph == "timeseries":
                timeseries_result=timeseries(df)
            elif  typegraph  == "distribution":
                distribution_result=distribution_page(df)
            elif  typegraph  == "scatter":
                scatter_result = scatter_page(df)
            elif  typegraph  == "boxplot":
                boxplot_result = boxplot_page(df)
            elif  typegraph  == "violine":
                violine_result = violine_page(df)

            # listtype = []
            st.caption("add graph to show")
            if st.button("Add to Plot"):
                
                listtype = None  # กำหนดค่าเริ่มต้นให้ x เป็น None
                if  typegraph == "timeseries":
                    data = typegraph
                    listtype = timeseries_result
                elif  typegraph  == "distribution":
                    data = typegraph
                    listtype = distribution_result
                elif  typegraph  == "scatter":
                    data = typegraph
                    listtype = scatter_result
                elif  typegraph  == "boxplot":
                    data = typegraph
                    listtype = boxplot_result
                elif  typegraph  == "violine":
                    data = typegraph
                    listtype = violine_result

                if listtype is not None:
                    st.session_state.x_array.append(listtype)
                    # st.write("listtype:", st.session_state.x_array)
                    st.session_state.data_choices.append(data)
                    # st.write("Data Choices:", st.session_state.data_choices)

            st.caption("show graph")
            if st.button("Plot Graph"):
                num_data_choices = len(st.session_state.data_choices)
                for i in range(0, num_data_choices, 4):
                    # Create a 2x2 grid of subplots
                    fig = go.Figure()
                    specs = [[{'type': 'scatter'}]*2 for _ in range(2)]
                    fig = make_subplots(rows=2, cols=2, subplot_titles=["", "", "", ""], specs=specs)

                    for j, data in enumerate(st.session_state.data_choices[i:i+4]):
                        row, col = divmod(j, 2)  # Calculate row and column indices
                        subplot_idx = row * 2 + col + 1
                        selected_data = st.session_state.data_choices
                        selected_array = st.session_state.x_array

                        if selected_data[j] == "timeseries":
                            dftime = selected_array[j]
                            print(dftime)
                            for column in dftime.columns[0:]:
                                fig.add_trace(go.Scatter(x=dftime.index, y=dftime[column], mode='lines',text= column, name='Timseries '+column),
                                            row=row+1, col=col+1)
                    
                            # Set subplot titles and axis labels
                            fig.update_layout(height=800,width=800)
                            fig.update_xaxes(title_text='TimeStamp',title_font=dict(size=10), row=row+1, col=col+1)
                            fig.update_yaxes(title_text='Value',title_font=dict(size=10), row=row+1, col=col+1)

                        elif selected_data[j] == "distribution":
                            dx_values = selected_array[j]
                            print(dx_values)
                            for col_idx, column in enumerate(dx_values.columns):

                                if column != "Class":
                                    for class_value in dx_values['Class'].unique():
                                        df_class_subset = dx_values[dx_values['Class'] == class_value]
                                        fig.add_trace(go.Histogram(x=df_class_subset[column], name=f"{column} - {class_value}", nbinsx=8),
                                                            row=row+1, col=col+1)
                                
                            # Set subplot titles and axis labels
                            fig.update_layout(height=800,width=800)
                            fig.update_xaxes(title_text = dx_values.columns[0] ,title_font=dict(size=10), row=row+1, col=col+1)
                            fig.update_yaxes(title_text='frequency',title_font=dict(size=10), row=row+1, col=col+1)

                        elif selected_data[j] == "scatter":
                            scatter_result = selected_array[j]

                            x_values = scatter_result.iloc[:, 0]
                            y_values = scatter_result.iloc[:, 1]
                            x_name = scatter_result.columns[0]
                            y_name = scatter_result.columns[1]

                            color_sequence = ['#3366FF','#FF5733' ]
                            mapping = {class_name: color for class_name, color in zip(scatter_result['Class'].unique(), color_sequence)}
                            df_mapping = pd.DataFrame(list(mapping.items()), columns=['Class', 'Color'])

                            fig.add_trace(go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='markers',
                                marker=dict(
                                    color=scatter_result['Class'].map(df_mapping.set_index('Class')['Color'])
                                ),
                                text= scatter_result['Class'],name = 'Scatter'

                            ), row=row+1, col=col+1)

                            fig.update_layout(height=800,width=800)
                            fig.update_xaxes(title_text=x_name, title_font=dict(size=10))
                            fig.update_yaxes(title_text=y_name, title_font=dict(size=10))

                        elif selected_data[j] == "boxplot":
                            df_boxplot=selected_array[j]
                            for col_idx, column in enumerate(df_boxplot.columns):
                            # ถ้าคอลัมน์ไม่ใช่ "Class"
                                if column != "Class":
                                    # เพิ่ม Box Plot แบบแยกสีตามคอลัมน์ "Class"
                                    for class_value in df_boxplot['Class'].unique():
                                        df_subset = df_boxplot[df_boxplot['Class'] == class_value]
                                        fig.add_trace(go.Box(y=df_subset[column], name=f"{column} - {class_value}", boxpoints=False, jitter=0.2), row=row+1, col=col+1)

                            
                            fig.update_layout(height=800,width=800)
                            fig.update_xaxes(visible=False)
                            fig.update_yaxes(title_text='Value')
                            

                        elif selected_data[j] == "violine":
                            df_violine=selected_array[j]

                            for col_idx, column in enumerate(df_violine.columns):
                            # ถ้าคอลัมน์ไม่ใช่ "Class"
                                if column != "Class":
                                    # เพิ่ม Box Plot แบบแยกสีตามคอลัมน์ "Class"
                                    for class_value in df_violine['Class'].unique():
                                        df_subset = df_violine[df_violine['Class'] == class_value]
                                        fig.add_trace(go.Violin(y=df_subset[column], name=f"{column} - {class_value}", points=False), row=row+1, col=col+1)

                            
                            fig.update_layout(height=800,width=800)
                            fig.update_xaxes(visible=False)
                            fig.update_yaxes(title_text='Value')

                    for j in range(len(st.session_state.data_choices[i:i+4]), 4):
                        row, col = divmod(j, 2)  # Calculate row and column indices
                        subplot_idx = row * 2 + col + 1
                        fig['layout']['xaxis' + str(subplot_idx)].visible = False
                        fig['layout']['yaxis' + str(subplot_idx)].visible = False
                        
                    # Show the plot
                    fig.update_layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ))
                    st.plotly_chart(fig, use_container_width=True,theme=None)

    elif  typetoplot == "specialty":
        typemodel = st.selectbox("Select types:", ["select plot","radar","3D scatter","joint plot(histogram)"])
        if typemodel == "select plot":
            st.warning("Please select plot")

        elif typemodel == "radar":
            radar_result = radar_page(df)

            if st.button("Plot"):
                df1 = radar_result
                # print(df1)
                df2 = df1.set_index('index')
                print(df2)
                # Assuming you have your data in the DataFrame df1
                # Define categories and angles
                categories = list(df1.columns[1:])
                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]

                # Define data for min and max
                values_min = df1.loc[0, categories].values.tolist()
                values_min += values_min[:1]

                values_max = df1.loc[1, categories].values.tolist()
                values_max += values_max[:1]

                # Create a Plotly figure
                fig = go.Figure()

                # Add traces for min and max
                fig.add_trace(go.Scatterpolar(
                    r=values_min,
                    theta=categories,
                    fill='toself',
                    name='min',
                    line=dict(color='blue'),
                ))

                fig.add_trace(go.Scatterpolar(
                    r=values_max,
                    theta=categories,
                    fill='toself',
                    name='max',
                    line=dict(color='red'),
                ))

                # Layout settings
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[values_min, values_max]
                        ),
                    ),
                )

                # Add labels and legend
                fig.update_layout(
                    showlegend=True,
                )
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))

                # Display in Streamlit
                st.plotly_chart(fig, theme=None)

        elif typemodel == "3D scatter":
            scatter3d_result = scatter3d_page(df)
            if scatter3d_result is None:
                print(scatter3d_result)
            else:
                x_values = scatter3d_result.iloc[:, 0]
                y_values = scatter3d_result.iloc[:, 1]
                z_values = scatter3d_result.iloc[:, 2]
                
                if st.button("plot"):
                    color_sequence = ['#3366FF','#FF5733' ]  # ตัวอย่างสีที่แตกต่างกัน  
                    fig = px.scatter_3d(scatter3d_result, x=x_values, y=y_values, z=z_values, 
                                        color='Class',
                                        color_discrete_sequence=color_sequence,
                                        labels={'x': scatter3d_result.columns[0], 
                                                'y': scatter3d_result.columns[1], 
                                                'z': scatter3d_result.columns[2]},
                                        width= 800, height= 800)
                    fig.update_layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ))
                    st.plotly_chart(fig, theme=None)

        elif typemodel == "joint plot(histogram)":
            jointpoint_result = jointpoint(df)
            print(jointpoint_result)
            if jointpoint_result is None:
                print(jointpoint_result)
            else:
                x_column = jointpoint_result.columns[0]
                y_column = jointpoint_result.columns[1]
                print(x_column,y_column)

                
                
                if st.button("plot"):

                    print(jointpoint_result)

                    fig = px.scatter(jointpoint_result, x=x_column, y=y_column, color="Class", 
                        marginal_x="histogram", marginal_y="histogram",
                        title="Scatter with Histogram")
                    
                    fig.update_layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ))

                    
                    st.plotly_chart(fig, theme=None)

        # elif typemodel == "joint plot(KDE)":
        #     jointpoint_result = jointpoint(df)
        #     # print(jointpoint_result)
        #     if jointpoint_result is None:
        #         print(jointpoint_result)
        #     else:
        #         unique_classes = jointpoint_result['Class'].unique()

        #         df1 = jointpoint_result[jointpoint_result['Class'] == unique_classes[0]]
        #         df2 = jointpoint_result[jointpoint_result['Class'] == unique_classes[1]]

        #         x_column = jointpoint_result.columns[0]
        #         y_column = jointpoint_result.columns[1]
        #         print(x_column,y_column)

        #         if st.button("plot"):
                    
        #             fig = go.Figure()

        #             # Add the first set of data with a specific color
        #             fig.add_trace(go.Histogram2dContour(
        #                 x=df1[x_column], y=df1[y_column], 
        #                 nbinsx=30, nbinsy=30,
        #                 colorscale='Blues',
        #                 xaxis='x', yaxis='y',
        #                 name=unique_classes[0]
        #             ))

        #             # Add the second set of data with a different color
        #             fig.add_trace(go.Histogram2dContour(
        #                 x=df2[x_column], y=df2[y_column], 
        #                 nbinsx=30, nbinsy=30,
        #                 colorscale='Reds',  # Change the color scale for the second set of data
        #                 xaxis='x', yaxis='y',
        #                 name=unique_classes[1]
        #             ))

        #             fig.add_trace(go.Scatter(
        #                 x=df1[x_column], y=df1[y_column], 
        #                 xaxis='x',
        #                 yaxis='y',
        #                 mode='markers',
        #                 marker=dict(
        #                     color='rgba(0,0,0,0.3)',
        #                     size=3
        #                 ),
        #                 name=unique_classes[0]
        #             ))

        #             fig.add_trace(go.Scatter(
        #                x=df2[x_column], y=df2[y_column], 
        #                 xaxis='x',
        #                 yaxis='y',
        #                 mode='markers',
        #                 marker=dict(
        #                     color='rgba(1,0,0,0.3)',
        #                     size=3
        #                 ),
        #                 name=unique_classes[1]
        #             ))

        #             # Side dist for the first set of data
        #             fig.add_trace(go.Histogram(
        #                 y=df1[y_column], 
        #                 xaxis='x2',
        #                 marker=dict(
        #                     color='rgba(0,0,0,1)'
        #                 ),
        #                 name=unique_classes[0]
        #             ))

        #             # Side dist for the second set of data
        #             fig.add_trace(go.Histogram(
        #                 x=df1[x_column],
        #                 yaxis='y2',
        #                 marker=dict(
        #                     color='rgba(0,0,0,1)'
        #                 ),
        #                 name=unique_classes[1]
        #             ))
        #             # Side dist for the first set of data
        #             fig.add_trace(go.Histogram(
        #                 y=df2[y_column], 
        #                 xaxis='x2',
        #                 marker=dict(
        #                     color='rgba(1,0,0,1)'
        #                 ),
        #                 name=unique_classes[0]
        #             ))

        #             # Side dist for the second set of data
        #             fig.add_trace(go.Histogram(
        #                 x=df2[x_column],
        #                 yaxis='y2',
        #                 marker=dict(
        #                     color='rgba(1,0,0,1)'
        #                 ),
        #                 name=unique_classes[1]
        #             ))

        #             fig.update_layout(
        #                 autosize=False,
        #                 xaxis=dict(
        #                     zeroline=False,
        #                     domain=[0, 0.85],
        #                     showgrid=False
        #                 ),
        #                 yaxis=dict(
        #                     zeroline=False,
        #                     domain=[0, 0.85],
        #                     showgrid=False
        #                 ),
        #                 xaxis2=dict(
        #                     zeroline=False,
        #                     domain=[0.85, 1],
        #                     showgrid=False
        #                 ),
        #                 yaxis2=dict(
        #                     zeroline=False,
        #                     domain=[0.85, 1],
        #                     showgrid=False
        #                 ),
        #                 height=600,
        #                 width=600,
        #                 bargap=0,
        #                 hovermode='closest',
        #                 showlegend=True,  # Set to True to display legends
        #                 legend=dict(x=0.9, y=1)  # Adjust legend position
        #             )

        #             # fig.show()
        #             st.plotly_chart(fig)
                
#-----------------------------------------------------------------------------------------------------------------------

def selecttypes_page():

    st.warning("please select type to plot at sidebar")

#------------------------------------------------------------------------------------------------------------------------
def distribution_page(df):  #1 class

    st.subheader("distribution page")
    
    df1 = pd.DataFrame(df[0])
    df2 = pd.DataFrame(df[1])

    dataset = pd.concat([df1, df2], axis=0, ignore_index=True)

    if dataset is not None :
        cols = [col for col in dataset.columns if col != "Class"]
        dx_column=st.selectbox("Select sensor:", cols )
        df_subset = dataset[[dx_column, "Class"]]
       
        df_subset = pd.DataFrame(df_subset).reset_index(drop=True)
        return df_subset
        
    
    else:
        st.warning("please select dataset")

#-------------------------------------------------------------------------------------------------------------------------

def timeseries(df): #1 class , multiparameter
    st.subheader("timeseries page")
    df1 = pd.DataFrame(df[0])
    df2 = pd.DataFrame(df[1])
    df = pd.concat([df1, df2])
    duplicates = df['Class'][df['Class'].duplicated(keep=False)].unique()
    # แสดงตัวเลือกใน st.selectbox
    selected_option = st.selectbox('selectdataset', duplicates)

    dataset = df[df['Class'] == selected_option]

    if dataset is not None :

        cols = [col for col in dataset.columns if col != "Class"]
        dx_column=st.multiselect("Select sensor:", cols )

        df_subset = dataset[dx_column]
        df_subset = pd.DataFrame(df_subset)
        print(df_subset)
        return df_subset
    
    else:
        st.warning("please select dataset")
#--------------------------------------------------------------------------------------------------------------------------
def scatter_page(df): #2 class , 2 parameter
    st.subheader("scatter page")
    
    df1 = pd.DataFrame(df[0])
    df2 = pd.DataFrame(df[1])

    dataset = pd.concat([df1, df2], axis=0, ignore_index=True)

   
    print(dataset)
    if dataset is not None :

        cols = [col for col in dataset.columns if col != "Class"]
        col_X=st.selectbox("Select sensorX:", cols )
        col_Y=st.selectbox("Select sensorY:", cols )
        
        df_subset = dataset[[col_X, col_Y, "Class"]]
        df_subset = pd.DataFrame(df_subset).reset_index(drop=True)
        return df_subset
    else:
        st.warning("please select dataset")

#--------------------------------------------------------------------------------------------------------------------------
def boxplot_page(df): #2 class , multiparameter
    st.subheader("boxplot page")
    
    df1 = pd.DataFrame(df[0])
    df2 = pd.DataFrame(df[1])

    dataset = pd.concat([df1, df2], axis=0, ignore_index=True)


    if dataset is not None :
        cols = [col for col in dataset.columns if col != "Class"]
        dx_column=st.multiselect("Select sensor:", cols )
        if dx_column != [] :
            dx_column = dx_column + ["Class"]
            df_subset = dataset[dx_column]
            df_subset = pd.DataFrame(df_subset).reset_index(drop=True)
            print(df_subset)
            return df_subset
        
    else:
        st.warning("please select dataset")
#--------------------------------------------------------------------------------------------------------------------------
def violine_page(df): #2 class , multiparameter
    st.subheader("violine page")
    
    df1 = pd.DataFrame(df[0])
    df2 = pd.DataFrame(df[1])


    dataset = pd.concat([df1, df2], axis=0, ignore_index=True)


    if dataset is not None :
        cols = [col for col in dataset.columns if col != "Class"]
        dx_column=st.multiselect("Select sensor:", cols )
        if dx_column != [] :
            dx_column = dx_column + ["Class"]
            df_subset = dataset[dx_column]
            df_subset = pd.DataFrame(df_subset).reset_index(drop=True)
            print(df_subset)
            return df_subset
        
    else:
        st.warning("please select dataset")

#--------------------------------------------------------------------------------------------------------------------------
def radar_page(df): #1 class , multiparameter
    st.subheader("radar page")
    df1 = pd.DataFrame(df[0])
    df2 = pd.DataFrame(df[1])

    df = pd.concat([df1, df2], ignore_index=True)

    duplicates = df['Class'][df['Class'].duplicated(keep=False)].unique()
    # แสดงตัวเลือกใน st.selectbox
    selected_option = st.selectbox('selectdataset', duplicates)

    dataset = df[df['Class'] == selected_option]
    

    if dataset is not None :

        cols = [col for col in dataset.columns if col != "Class"]
        dx_column=st.multiselect("Select sensor:", cols )
        if dx_column != [] :
            df_subset = dataset[dx_column ]
            df_subset = pd.DataFrame(df_subset).reset_index(drop=True)
            min_values = df_subset.min()
            max_values = df_subset.max()
            
            min_series = pd.Series(min_values, name='min')
            max_series = pd.Series(max_values, name='max')
            
            result_df = pd.concat([min_series, max_series], axis=1)
            # result_df['SensorName'] = selected_columns
            # result_df = result_df[['SensorName', 'min', 'max']]
            # df1_transposed = result_df.T # or df1.transpose()
            
            result_df_transposed = result_df.transpose().reset_index()
            st.write(result_df_transposed)
            df1 = result_df_transposed
            print(df1)
            return df1
    else:
        st.warning("please select dataset")

#--------------------------------------------------------------------------------------------------------------------------
def scatter3d_page(df): #2 class , 3 parameter
    st.subheader("scatter page")
    
    dataset1 = pd.DataFrame(df[0])
    dataset2 = pd.DataFrame(df[1])
    dataset = pd.concat([dataset1, dataset2], axis=0, ignore_index=True)
    print(dataset)
    if dataset is not None :

        cols = [col for col in dataset.columns if col != "Class"]
        col_X=st.selectbox("Select sensorX:", cols )
        col_Y=st.selectbox("Select sensorY:", cols )
        col_z=st.selectbox("Select sensorZ", cols )
        
        df_subset = dataset[[col_X, col_Y,col_z, "Class"]]
        df_subset = pd.DataFrame(df_subset).reset_index(drop=True)
        return df_subset
    else:
        st.warning("please select dataset")
#--------------------------------------------------------------------------------------------------------------------------
def jointpoint(df): #2 class , 3 parameter
    st.subheader("jointplot(KDE) page")
    
    dataset1 = pd.DataFrame(df[0])
    dataset2 = pd.DataFrame(df[1])
    dataset = pd.concat([dataset1, dataset2], axis=0, ignore_index=True)
    print(dataset)
    if dataset is not None :

        cols = [col for col in dataset.columns if col != "Class"]
        col_X=st.selectbox("Select sensorX:", cols )
        col_Y=st.selectbox("Select sensorY:", cols )
        
        df_subset = dataset[[col_X, col_Y, "Class"]]
        df_subset = pd.DataFrame(df_subset).reset_index(drop=True)
        return df_subset
    else:
        st.warning("please select dataset")

#--------------------------------------------------------------------------------------------------------------------------
def timespanforecast(df_point,token):
    selected_values = st.selectbox('select point_id:', df_point['names'])
    # print(selected_valuesX)  #['SCG.PP3IP21.JI0401P', 'SCG.PP3IP21.KE40401', 'SCG.PP3IP21.TI0410', 'SCG.PP3IP21.TI40404P']
    point_ids = df_point[df_point['names'] == selected_values]['point_ids'].values[0]

    st.session_state.point_ids = point_ids

    # st.checkbox(df_point)
    current_datetime = datetime.datetime.now()
    # print(type)
    # ลบ 3 ปีจากวันที่และเวลาปัจจุบัน
    start = current_datetime - datetime.timedelta(days=365*2)
    # ลบ 12 ชั่วโมงจากวันที่และเวลาปัจจุบัน

    end = current_datetime - datetime.timedelta(days=1)
    # print(start,end)

    col1, col2 , col3 = st.columns(3)

    with col1:

        start_date1 = st.date_input("select start date1", end.date(),start.date(),end.date())
        start_time1 = st.time_input("select start time1", datetime.time(0, 0))
        
    with col2:
        # เลือกเวลาสิ้นสุด
        end_date1 = st.date_input("select end date1", end.date(),start.date(),end.date())
        end_time1 = st.time_input("select end time1", datetime.time(0, 0))

    with col3:
        frequency = st.number_input('frequency (mins)', min_value=5, max_value=30, value=5)
        daytoprediction = st.number_input('Day to Prediction', min_value=1, max_value=31, value=1)
    
    start_datetime1 = pd.Timestamp(datetime.datetime.combine(start_date1, start_time1))
    end_datetime1 = pd.Timestamp(datetime.datetime.combine(end_date1, end_time1))
    

    start_datetime1 = start_datetime1.to_pydatetime()
    end_datetime1 = end_datetime1.to_pydatetime()
    

    if start_datetime1 >= end_datetime1:
        st.error("please select new period 1.")
    else: 
        if frequency is not None:
            if frequency == "5 minute":
                frequency = 5
            elif frequency == "10 minute":
                frequency = 10
            elif frequency == "15 minute":
                frequency = 10
            
            period = [start_datetime1, end_datetime1]
            data = histaricalforcast(frequency,point_ids,period,token)
            if data !=[]:
                
                df_list1 = []

                for item in data:
                    point_name = item["PointName"]

                    for data_point in item["DataPoints"]:
                        timestamp = data_point["TimeStamp"]
                        value = data_point["Value"]
                        df_list1.append({"PointName": point_name, "TimeStamp": timestamp, "Value": value})

                # สร้าง DataFrame จากรายการข้อมูล
                df = pd.DataFrame(df_list1)
                # print(df)

                df = df.pivot(index='TimeStamp', columns='PointName', values='Value')
                df = df.reset_index()
                name = selected_values
                value = df.iloc[:, 0]
                # print(name)
                return name,df,daytoprediction,frequency
                
            else: 
                st.warning("please selected timespan")
    
#-------------------------------------------------------------------------------------------------------------------------------
def histaricalforcast(frequency,point_ids,period,token):

    point_ids = np.array(point_ids)
    point_ids = point_ids.tolist()

    if period is not None:

        starttime = period[0]
        starttime = starttime.strftime("%m/%Y/%d %H:%M:%S")

        endtime = period[1]
        endtime = endtime.strftime("%m/%Y/%d %H:%M:%S")
        #-------------------------------------------------------------------------
        url = "https://repcrydrpws71/api/v1/histdata/"

        # สร้าง payload โดยแทนค่า "PointIDs" ด้วย project_ids
        payload = json.dumps({
        "PointIDs": [point_ids],
        "StartDateTime": starttime,
        "EndDateTime": endtime,
        "FrequencySeconds": frequency * 60  #15 min
        })

        headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+token }

        response = requests.request("POST", url, headers=headers, data=payload,verify=False)
        response = response.text
        response = json.loads(response)
        if "ErrorDetails" in response:
            st.error(response["ErrorDetails"])
        else:
            return response
#-------------------------------------------------------------------------------------------------    
def linear(name,df,day,frequen):
    print(df)
    train_size = int(len(df))
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    last_timestamp = df['TimeStamp'].iloc[len(df) - 1]
    new_timestamps = pd.date_range(last_timestamp, periods=day*24*60//frequen, freq=str(frequen) + 'T')
    new_data = {
    'TimeStamp': new_timestamps,
    }
    df_new = pd.DataFrame(new_data)
    # รวม DataFrame ใหม่กับ DataFrame เดิม
    df = pd.concat([df, df_new])
    df.set_index('TimeStamp', inplace=True)
    

    df = pd.DataFrame(df)
    train_data = df[:train_size]
    # สร้างคุณลักษณะ (features)
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data.iloc[:,0].values
    # สร้างและฝึกตัวแบบ Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = np.arange(0, len(df)).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    # # คำนวณขอบเขตคาดการณ์
    # prediction_lower = y_pred - 0.1  # ตั้งค่าต่ำสุดของขอบเขตคาดการณ์
    # prediction_upper = y_pred + 0.1  # ตั้งค่าสูงสุดขอบเขตคาดการณ์

    fig = go.Figure()
    # Plot training data
    fig.add_trace(go.Scatter(x=train_data.index, y=y_train, mode='lines', name='Training Data', line=dict(color='blue')))
    # Plot test data
    # Plot predicted data
    fig.add_trace(go.Scatter(x=df.index, y=y_pred, mode='lines', name='Predicted Data', line=dict(color='red')))
    # fig.add_trace(go.Scatter(x=df.index, y=prediction_lower, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', mode='none', name='Lower'))
    # fig.add_trace(go.Scatter(x=df.index, y=prediction_upper, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', mode='none', name='Upper'))
    fig.update_layout(
        title=' Forecasting with Linear Regression Prediction',
        xaxis_title='TimeStamp',
        yaxis_title=name,
    )
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=90)
    # Display legend
    fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        ))
    st.plotly_chart(fig, theme=None)
#-------------------------------------------------------------------------------------------------    
def quadractic(name,df,day,frequen):
    print(df)
    train_size = int(len(df))
    # แปลงคอลัมน์ 'TimeStamp' เป็น datetime
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    last_timestamp = df['TimeStamp'].iloc[len(df) - 1]
    new_timestamps = pd.date_range(last_timestamp, periods=day*24*60//frequen, freq=str(frequen) + 'T')
    # สร้าง DataFrame ใหม่ที่มี timestamp ที่เพิ่มเข้าไป
    new_data = {
        'TimeStamp': new_timestamps,
    }
    

    df_new = pd.DataFrame(new_data)
    
    # รวม DataFrame ใหม่กับ DataFrame เดิม
    df = pd.concat([df, df_new])

    df = pd.DataFrame(df)

    df = df.set_index('TimeStamp')

    train_data = df[:train_size]
    test_data = df[train_size:]

    # สร้างคุณลักษณะ (features)
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data.iloc[:,0].values

    degree = 2
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    # สร้างคุณลักษณะ polynomial สำหรับข้อมูลทดสอบ
    X_test = np.arange(0, len(df)).reshape(-1, 1)
    # ทำการพยากรณ์ด้วยโมเดล Quadratic Regression
    y_pred = model.predict(X_test)
    # Create a Plotly figure to visualize the results
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(x=train_data.index, y=y_train, mode='lines', name='Training Data', line=dict(color='blue')))


    # Plot predicted data
    fig.add_trace(go.Scatter(x=df.index, y=y_pred, mode='lines', name='Predicted Data', line=dict(color='red')))

    # Update the layout of the plot
    fig.update_layout(
        title='Forecasting with Quadratic Regression Prediction',
        xaxis_title='TimeStamp',
        yaxis_title= name,
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=90)

    # Display the legend
    fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))  
    # Display the Plotly chart and the Mean Squared Error
    st.plotly_chart(fig, theme=None)
#-------------------------------------------------------------------------------------------------
def prophetforecast(name,df,day,frequen):
    # print(df)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['TimeStamp'] = df['TimeStamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # name = df.columns[1]
    df = df.rename(columns={'TimeStamp': 'ds',df.columns[1]: 'y'})
    
    m = Prophet()
    m.fit(df)

    # Python
    future = m.make_future_dataframe(periods=day*24*60//frequen, freq=str(frequen) + 'T')
    
    forecast = m.predict(future) 
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # สร้างกราฟด้วย Plotly
    fig = go.Figure()

    # เพิ่มข้อมูลจริง ("y") และผลลัพธ์ที่ได้จาก Prophet ("yhat") เข้าสู่กราฟ
    fig.add_trace(go.Scatter(x=forecast['ds'], y=df['y'], mode='lines', name='actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='prediction', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(0,100,80,0.2)', mode='none', name='lower'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', fillcolor='rgba(0,100,80,0.2)', mode='none', name='upper'))

    # ปรับแต่งชื่อแกน X และ Y
    fig.update_xaxes(title_text="TimeStamp")
    fig.update_yaxes(title_text=name)

    # ... ปรับแต่งกราฟเพิ่มเติมตามที่คุณต้องการ ...
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ))
    # แสดงกราฟด้วย Plotly
    st.plotly_chart(fig, theme=None)
    

#-------------------------------------------------------------------------------------------------    
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
    
def neural(name,df,day,frequen):
    # fix random seed for reproducibility
    df = pd.DataFrame(df)
    np.random.seed(7)
    # load the dataset
    dataframe = df
    timestap = dataframe.iloc[:, 0]
    dataframe = dataframe.iloc[:, 1]
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim=look_back))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(dataset))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()
    # สร้างกราฟโดยใช้ Plotly.graph_objects

    # สร้าง DataFrame ตัวอย่างข้อมูล
    data = {'Timestamp': timestap,
          'Actual': scaler.inverse_transform(dataset).flatten(),
            # 'TrainPredict': trainPredictPlot.flatten(),
            'TestPredict': testPredictPlot.flatten()}
    df = pd.DataFrame(data)
    # print(df)
    df.set_index('Timestamp', inplace=True)

    # สร้างกราฟโดยใช้ Plotly Express
    fig = px.line(df, title="Forecasting LSTM(Neural Network)")
    
    st.plotly_chart(fig, theme=None)
    st.write(f"Mean Squared Error (MSE): {testScore}")
#-------------------------------------------------------------------------------------------------    
def timeforOMR(df_point,token):
    print(df_point)
    selected_values = st.selectbox('select point_id:', df_point['names'])
    # print(selected_valuesX)  #['SCG.PP3IP21.JI0401P', 'SCG.PP3IP21.KE40401', 'SCG.PP3IP21.TI0410', 'SCG.PP3IP21.TI40404P']
    point_ids = df_point[df_point['names'] == selected_values]['point_ids'].values[0]
    st.session_state.point_ids = point_ids

    # st.checkbox(df_point)
    current_datetime = datetime.datetime.now()
    # print(type)
    # ลบ 3 ปีจากวันที่และเวลาปัจจุบัน
    start = current_datetime - datetime.timedelta(days=365*2)
    # ลบ 12 ชั่วโมงจากวันที่และเวลาปัจจุบัน

    end = current_datetime - datetime.timedelta(days=1)
    # print(start,end)

    col1, col2,col3 = st.columns(3)

    with col1:

        start_date1 = st.date_input("select start date1", end.date(),start.date(),end.date())
        start_time1 = st.time_input("select start time1", datetime.time(0, 0))
        
        
    with col2:
        # เลือกเวลาสิ้นสุด
        end_date1 = st.date_input("select end date1", end.date(),start.date(),end.date()) 
        end_time1 = st.time_input("select end time1", datetime.time(0, 0))

    with col3:
        frequency = st.radio(
            "frequecy datapoint",
            ["5 minute", "10 minute", "15 minute", "20 minute", "25 minute"],
            index=None,
        )
    
    start_datetime1 = pd.Timestamp(datetime.datetime.combine(start_date1, start_time1))
    end_datetime1 = pd.Timestamp(datetime.datetime.combine(end_date1, end_time1))
    

    start_datetime1 = start_datetime1.to_pydatetime()
    end_datetime1 = end_datetime1.to_pydatetime()
    

    
    if start_datetime1 >= end_datetime1:
        st.error("please select new period 1.")
    else: 
        period = [start_datetime1, end_datetime1]
        if st.button("submit"):

            if frequency is not None:
                if frequency == "5 minute":
                    frequency = 5
                elif frequency == "10 minute":
                    frequency = 10
                elif frequency == "15 minute":
                    frequency = 10
                elif frequency == "20 minute":
                    frequency = 20
                elif frequency == "25 minute":
                    frequency = 25

                data = prediction(frequency,point_ids,period,token)


                df = pd.DataFrame()
                for item in data:
                    for output in item['PointOutputs']:
                        if output['pointTypeId'] == 11:
                            df.loc[output['x'], 'actual'] = output['y']
                        elif output['pointTypeId'] == 12:
                            df.loc[output['x'], 'predict'] = output['y']

                df.index = pd.to_datetime(df.index)
                if df is not None:
                    return selected_values,df
                else: 
                    st.warning("please selected timespan")
            else:
                st.warning("please select frequency datapoint")
        else:
            st.warning("please confirm timespan")

#--------------------------------------------------------------------------------------------------------------------------
def prediction(frequency,point_ids,period,token):
    point_ids = np.array(point_ids)
    point_ids = point_ids.tolist()
    # print(point_ids)
    if period is not None:

        starttime = period[0]
        starttime = starttime.strftime("%m/%Y/%d %H:%M:%S")

        endtime = period[1]
        endtime = endtime.strftime("%m/%Y/%d %H:%M:%S")

        url = "https://repcrydrpws71/api/web/trend/historical"

        payload = json.dumps({
        "DataSource": "PrismArchive","StartDateTime": starttime,"EndDateTime": endtime,
        "FrequencySeconds": frequency*60,"InputPointIds":[point_ids],"OutputPointTypeIds": [11,12]})
        headers = {
        'Content-Type': 'application/json','Authorization': 'Bearer '+token
        }

        response = requests.request("POST", url, headers=headers, data=payload,verify=False)
        response = response.text
        response = json.loads(response)
        if "ErrorDetails" in response:
            st.error(response["ErrorDetails"])
        else:
            return response
        
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
   home()