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
    data = np.array(data).reshape(row_count, 3)
    obj = PredictionPipeline()
    predict = obj.predict(data)
    my_result=pd.DataFrame(predict)
    #print(my_result.head())
    impacted_servers = (my_result.iloc[:, 0] > 90).sum()
    print(f"impacted_servers={impacted_servers}")
    increased_load_data=data.merge(impacted_servers,left_index=True, right_index=True)

    increased_load_data.to_csv('merged_data.csv', index=False)

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
    st.session_state['current_page'] = 'home'

def navigate(page):
    st.session_state['current_page'] = page

def home_page():
    os.system("python main.py")
    about_text="""
        In modern enterprise, the <b>complexity of IT systems</b> based on interdependency between 
        several components, makes it <b>extremely difficult</b> to explain behavior and <b>predict the outcome</b> 
        when a modification occurs. System administrators often struggle to 
        anticipate the effects of <b>workload fluctuations</b> or <b>infrastructure modifications</b>, 
        leading to inefficiency and high operational costs. This project develops a 
        <b>'what-if' analysis tool</b> to model potential changes and assess their 
        impact on system performance and resource utilization, enabling more 
        <b>informed decision-making</b> and reducing risks in <b>dynamic IT environments</b>.<br>"""

    approch_text="""This project, <b>"SysForecast"</b>, employs a sophisticated approach by integrating
      machine learning algorithms with <b>Elasticsearch</b> to predict the impact of <b>hypothetical changes</b>
        in IT enterprise systems. The project is structured into a <b>modular pipeline</b>, ensuring 
        <b>scalability</b> and <b>maintainability</b>. I leverage the robustness of Elasticsearch for data 
        ingestion and <b>XGBoost and SVM</b> (Support Vector Machines) for predictive modeling, 
        facilitating an in-depth <b>what-if analysis</b><br>.
        """
    
    methodology_text="""The project is organized into a sequential pipeline, outlined as follows:<br>
<u><b>Data Ingestion:</b>Stage 01</u> begins with data ingestion, utilizing Elasticsearch for its powerful full-text
search capabilities and speed, allowing us to handle large volumes of IT system data efficiently.<br>
<u><b>Data Validation:</b>In Stage 02</u>, the ingested data is validated against predefined schemas to ensure quality and consistency, 
which is crucial for reliable predictions.<br>
<u><b>Data Transformation:</b>Stage 03</u> involves data transformation where raw data is cleaned and converted into a 
format suitable for machine learning algorithms.<br>
<u><b>Model Training:</b>During Stage 04</u>, we train two machine learning models—SVM and XGBoost. 
SVM is known for its effectiveness in high-dimensional spaces, while XGBoost provides a gradient boosting framework that 
is widely used in winning Kaggle competitions.<br>
<u><b>Model Evaluation:</b>In Stage 05</u>, the performance of these models is evaluated using metrics such as 
accuracy, precision, recall, and F1-score to ensure they can make reliable predictions.<br>
<u><b>Prediction:</b> Finally, Stage 06</u> leverages the trained models to predict the outcomes of various what-if scenarios, 
helping system administrators make informed decisions.<br><br>
The pipeline components are built within a <b>Python-based framework</b>, ensuring a seamless flow from data ingestion to prediction.<br>
Additionally, a <b>Streamlit</b> application is developed for easy interaction with the model, enabling users to <b>visualize</b> 
the potential <b>impact</b> of changes in the IT infrastructure.<br><br>"""
    
    st.subheader("Problem Statement:")
    st.markdown(f"<div style='text-align: justify;'>{about_text}</div>", unsafe_allow_html=True)
    st.subheader("Approch:")
    st.markdown(f"<div style='text-align: justify;'>{approch_text}</div>", unsafe_allow_html=True)
    st.subheader("Methodology:")
    st.markdown(f"<div style='text-align: justify;'>{methodology_text}</div>", unsafe_allow_html=True)
    #st.sidebar.button('Predictions')
    
def pred_page():
    st.title('Predictions')
    st.subheader("Current State of IT enterprise:")
    # Input fields for the user to enter data
    # Adjust these inputs based on what your model expects
    #data=pd.read_csv("artifacts/data_ingestion/allmetrics.csv")
    csv_file_path=('static/current_it_data/allmetrics.csv')
    fig = plot_metric_distribution(csv_file_path)
    st.pyplot(fig)
    
    
    
    input_data = []
    input_data.append(st.number_input('Enter first feature value', value=0.0))
    #input_data.append(st.number_input('Enter second feature value', value=0.0))
    # Add more inputs as needed

    #if st.button('Train',help="First lets Tarin the model on Current Data"):
    #    st.write('Why hello there')
        # Predict button
    if st.button('Predict'):
        result = predict(input_data)
        st.write(f'The predicted output is: {result}')

def main():
    # Navigation bar
    project_title="""<h1><div style='text-align: center; color: #ff6961;'>
                        <span>SysForecast</span>
                    </div></h1><br><br>
                 """
    st.markdown(project_title, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("About"):
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