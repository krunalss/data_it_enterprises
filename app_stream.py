import os 
import numpy as np
import pandas as pd
from data_it_entp.pipeline.stage_06_prediction import PredictionPipeline
from data_it_entp.components.model_evaluation import ModelEvaluation
import streamlit as st 
import matplotlib.pyplot as plt


def predict(input_data):
    """ 
    Function to make predictions using the loaded model.
    Adjust this based on how your model processes input.
    """
    # Assuming model expects a list of features
    incresed_load =float(input_data[0])
    df = pd.read_csv('static/current_it_data/allmetrics.csv')      
    features = ['CPU_Load', 'Memory_Usage', 'Disk_Usage']


    data =df[features]*incresed_load
    data =data.clip(upper=100)
    row_count=len(df)
    #data = np.array(data).reshape(row_count, 3)
    
    obj = PredictionPipeline()
    predict = obj.predict(data)
    
    my_result=pd.DataFrame(predict)
    #my_result_df = pd.DataFrame(my_result)
    my_result_df = pd.DataFrame(predict, columns=['Status'])

    #print(my_result.head())
    impacted_servers = (my_result.iloc[:, 0] > 90).sum()
    print(f"impacted_servers={impacted_servers}")
    
    increased_load_data=pd.concat([data, my_result_df], axis=1)
    
    directory = 'static/predicted_data'
    file_path = os.path.join(directory, 'merged_data.csv')
    increased_load_data.to_csv(file_path, index=False)

    return impacted_servers

def plot_metric_distribution(csv_file_path):
    # Load the dataset
    data = pd.read_csv(csv_file_path)

    # Define the function to categorize the values
    def categorize_values(value):
        if value <= 25:
            return 'Low (<=25)'
        elif value > 75:
            return 'High (>75)'
        else:
            return 'Normal (26-75)'

    # Apply the function to the columns
    data['CPU_Load_Category'] = data['CPU_Load'].apply(categorize_values)
    data['Memory_Usage_Category'] = data['Memory_Usage'].apply(categorize_values)
    data['Disk_Usage_Category'] = data['Disk_Usage'].apply(categorize_values)
    data['Status_Category'] = data['Status'].apply(categorize_values)

    # Calculate the counts for each category
    cpu_counts = data['CPU_Load_Category'].value_counts().reindex(['Low (<=25)', 'Normal (26-75)', 'High (>75)']).fillna(0)
    memory_counts = data['Memory_Usage_Category'].value_counts().reindex(['Low (<=25)', 'Normal (26-75)', 'High (>75)']).fillna(0)
    disk_counts = data['Disk_Usage_Category'].value_counts().reindex(['Low (<=25)', 'Normal (26-75)', 'High (>75)']).fillna(0)
    status_count=data['Status_Category'].value_counts().reindex(['Low (<=25)', 'Normal (26-75)', 'High (>75)']).fillna(0)

    # Stacked bar plot
    categories = ['CPU Load', 'Memory Usage', 'Disk Usage' ,"Query/Response Time"]
    values = np.array([cpu_counts, memory_counts, disk_counts,status_count])

    fig, ax = plt.subplots(figsize=(10, 6))

    # The bar locations on the x-axis
    bar_locations = np.arange(len(categories))

    # The width of the bars
    bar_width = 0.5

    # Colors for each section
    colors = ['green', 'blue', 'red']

    # Plotting each stack
    for i in range(len(colors)):
        ax.bar(bar_locations, values[:, i], bar_width, bottom=np.sum(values[:, :i], axis=1), color=colors[i], label=cpu_counts.index[i])

    # Adding the text labels for each bar
    for i in range(len(categories)):
        for j in range(len(colors)):
            height = np.sum(values[i, :j])
            ax.text(i, height + values[i, j]/2, f'{int(values[i, j])}', ha='center', va='center')

    # Setting the position of the x ticks
    ax.set_xticks(bar_locations)

    # Setting the labels for the x ticks
    ax.set_xticklabels(categories)

    # Adding the legend
    ax.legend(title='Category')

    # Setting the labels for the axes
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Number of Servers')

    return plt.gcf()

# Streamlit webpage title
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'prediction_project'
    #os.system("python main.py")

def navigate(page):
    st.session_state['current_page'] = page

def home_page():
    
    about_text="""
        In the complex landscape of <b> modern IT systems</b>, accurately predicting 
        server loads during high-demand periods such as <b>Black Friday sales, Great Amazon Sales, </b>
        and similar events is a significant challenge. These occasions often lead to unprecedented 
        traffic spikes, causing <b>server overloads and potential system downtimes.</b> 
        The interdependencies among various components in IT infrastructure make it difficult to 
        anticipate the effects of workload fluctuations or modifications, 
        often leading to inefficiency and high operational costs.<br>
        <b>SysForecast</b> addresses these challenges by offering a <b>'what-if' analysis tool</b>, 
        enabling system administrators to model potential changes and assess their impact on 
        system performance, resource utilization, and server load during peak times. 
        This tool is designed to facilitate <b>informed decision-making</b> and reduce risks 
        in dynamic IT environments.<br>"""

    approch_text="""
    <b>S</b>ysForecast uses <b>XGBoost</b>, a powerful machine learning algorithm, 
    to predict the impact of hypothetical changes in IT systems. Our approach centers on <b>MLflow</b>, 
    used for experiment tracking, model management, and facilitating reproducibility in the 
    machine learning workflow.
    The project is structured into a <b>modular pipeline</b>, ensuring scalability and maintainability. 
    With XGBoost's analytical prowess and MLflow's lifecycle management, 
    SysForecast provides detailed what-if analyses, aiding in strategic decision-making during critical 
    events like high-demand sales periods.
    This combination ensures <b>scalability, maintainability, and optimized performance</b> in dynamic IT environments.
    """
    
    methodology_text="""TThe SysForecast project follows a sequential pipeline comprising the following stages:

1. **Data Ingestion:** 
The Data Ingestion pipeline in this code efficiently handles data acquisition and preparation. 
It automates the downloading of data from a <b>specified API</b>, checking for pre-existing files to avoid redundancy, 
and then seamlessly extracts the contents of zip files into a designated directory, preparing the data for subsequent processing stages.

2. **Data Validation:** Ensuring the quality and consistency of ingested data against predefined <b>schemas</b>, which is crucial for reliable predictions.

3. **Data Transformation:** Cleaning and converting raw data into a format suitable for machine learning algorithms.

4. **Model Training:** Utilizes <b>XGBoost</b> for training the predictive model, renowned for its performance and accuracy.

5. **Model Evaluation:** Evaluates model performance using metrics such as <b>Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), 
and R-squared (R2)</b>. RMSE provides a measure of how accurately the model predicts the response, 
MAE gives an average magnitude of errors in a set of predictions, without considering their direction. 
R2, on the other hand, provides a measure of how well observed outcomes are replicated by the model, 
based on the proportion of total variation of outcomes explained by the model.

6. **Prediction:** Using the trained models to predict the outcomes of various what-if scenarios, assisting system administrators in making informed decisions.

Each component of the pipeline is built within a Python-based framework, ensuring a seamless flow from data ingestion to prediction.

## Streamlit Application
To enhance user interaction, a Streamlit application is developed, enabling users to easily interact with the model and visualize the potential impact of changes in IT infrastructure."""
    
    st.subheader("Problem Statement:")
    st.markdown(f"<div style='text-align: justify;'>{about_text}</div>", unsafe_allow_html=True)
    st.subheader("Approch:")
    st.markdown(f"<div style='text-align: justify;'>{approch_text}</div>", unsafe_allow_html=True)
    st.subheader("Methodology:")
    st.markdown(f"<div style='text-align: justify;'>{methodology_text}</div>", unsafe_allow_html=True)
    if st.button("Lets have a Sample Prediction Here"):
            navigate('prediction_project')
    
def pred_page():
    st.title('Predictions')
    st.subheader("Current State of IT enterprise(5000 server):")
    # Input fields for the user to enter data
    # Adjust these inputs based on what your model expects
    #data=pd.read_csv("artifacts/data_ingestion/allmetrics.csv")
    csv_file_path=('static/current_it_data/allmetrics.csv')
    fig = plot_metric_distribution(csv_file_path)
    st.pyplot(fig)
    
    
    
    input_data = []
    st.markdown(f"<h3 style='text-align: center'> Enter how much times load will incresed (for example on Black friday SALE/Great Amazon SALE )</h1>", unsafe_allow_html=True)
    input_data.append(st.number_input('2x,3x,4x', value=1))
    #input_data.append(st.number_input('Enter second feature value', value=0.0))
    # Add more inputs as needed

    #if st.button('Train',help="First lets Tarin the model on Current Data"):
    #    st.write('Why hello there')
        # Predict button
    if st.button('Predict'):
        result = predict(input_data)
        csv_file_path_result=('static/predicted_data/merged_data.csv')
        fig = plot_metric_distribution(csv_file_path_result)
        st.subheader("State of IT enterprise post increased of load")
        st.pyplot(fig)
        st.markdown(f"<h2 style='text-align: center'>Impacted servers count will be: {result}/5000 </h1>", unsafe_allow_html=True)
        

def main():
    # Navigation bar
    project_title="""<h1><div style='text-align: center; color: #ff6961;'>
                        <span>SysForecast</span>
                    </div></h1><br><br>
                 """
    st.markdown(project_title, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("About the Project"):
            navigate('home')

    with col2:
        if st.button("Prediction"):
            navigate('prediction_project')

    # Display the selected page
    if st.session_state['current_page'] == 'home':
        home_page()
    elif st.session_state['current_page'] == 'prediction_project':
        pred_page()

if __name__ == "__main__":
    main()