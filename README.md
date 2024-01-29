# SysForecast: What-If Analysis Tool for IT Enterprises

## Live Application
The application is deployed and can be accessed on Streamlit Cloud. 
Experience the live application here: [Streamlit App](https://dataitenterprises-whatif.streamlit.app/)  

Also for [MlFlow experimentation](https://dagshub.com/krunalss/data_it_enterprises.mlflow/)

## Problem Statement
In the complex landscape of modern IT systems, accurately predicting server loads during high-demand periods such as Black Friday sales, Great Amazon Sales, and similar events is a significant challenge. These occasions often lead to unprecedented traffic spikes, causing server overloads and potential system downtimes. The interdependencies among various components in IT infrastructure make it difficult to anticipate the effects of workload fluctuations or modifications, often leading to inefficiency and high operational costs. SysForecast addresses these challenges by offering a 'what-if' analysis tool, enabling system administrators to model potential changes and assess their impact on system performance, resource utilization, and server load during peak times. This tool is designed to facilitate informed decision-making and reduce risks in dynamic IT environments.

## Approach
SysForecast uses XGBoost, a powerful machine learning algorithm, to predict the impact of hypothetical changes in IT systems.Our approach centers on MLflow, used for experiment tracking, model management, and facilitating reproducibility in the machine learning workflow.The project is structured into a modular pipeline, ensuring scalability and maintainability. With XGBoost's analytical prowess and MLflow's lifecycle management, SysForecast provides detailed what-if analyses, aiding in strategic decision-making during critical events like high-demand sales periods. This combination ensures scalability, maintainability, and optimized performance in dynamic IT environments.

## Methodology
The SysForecast project follows a sequential pipeline comprising the following stages:

1. **Data Ingestion:** 
The Data Ingestion pipeline in this code efficiently handles data acquisition and preparation. It automates the downloading of data from a specified API, checking for pre-existing files to avoid redundancy, and then seamlessly extracts the contents of zip files into a designated directory, preparing the data for subsequent processing stages.

2. **Data Validation:** Ensuring the quality and consistency of ingested data against predefined schemas, which is crucial for reliable predictions.

3. **Data Transformation:** Cleaning and converting raw data into a format suitable for machine learning algorithms.

4. **Model Training:** Utilizes XGBoost for training the predictive model, renowned for its performance and accuracy.

5. **Model Evaluation:** Evaluates model performance using metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2). RMSE provides a measure of how accurately the model predicts the response, MAE gives an average magnitude of errors in a set of predictions, without considering their direction. R2, on the other hand, provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.

6. **Prediction:** Using the trained models to predict the outcomes of various what-if scenarios, assisting system administrators in making informed decisions.

Each component of the pipeline is built within a Python-based framework, ensuring a seamless flow from data ingestion to prediction.

## Streamlit Application
To enhance user interaction, a Streamlit application is developed, enabling users to easily interact with the model and visualize the potential impact of changes in IT infrastructure.

## Containerization with Docker
SysForecast is containerized using Docker, ensuring that the application runs seamlessly in any environment. The Dockerfile in the repository outlines the steps to create the Docker image, which packages the application along with its dependencies.

## CI/CD with GitHub Actions
The project utilizes GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD), automating the testing and deployment process. This ensures that every change made to the codebase is automatically tested and deployed to AWS, maintaining the reliability and stability of the application.

## Repository Structure
Below is an outline of the key directories and files in this repository, and their purpose:
- `artifacts/`: Contains the output files and data generated by the pipeline, including models, figures, and reports.
- `config/`: Houses configuration files used throughout the project to maintain consistency in settings and parameters.
- `data_it_entp/`: Core directory for the project's Python modules, including:
   -`components/`: Contains the individual Python scripts for each stage of the pipeline:
    - `data_ingestion.py`: Script for ingesting data into the system.
    - `data_transformation.py`: Handles the transformation of raw data for machine learning.
    - `data_validation.py`: Validates the data against predefined schemas.
    - `model_evaluation.py`: Evaluates the performance of the machine learning models.
    - `model_trainer.py`: Trains the SVM and XGBoost models.
  - `config/`: Configuration scripts for the components.
  - `constants/`: Defines constants used across the project.
  - `entity/`: Entities or data models used in the project.
  - `pipeline/`: Scripts that orchestrate the execution of the pipeline stages.
- `utils/`: Utility scripts providing common functions used by various components.
- `static/`: Static files required for the project, such as CSS for the Streamlit app.
- `templates/`: HTML templates if used for reporting or visualization.
- `Dockerfile`: Contains all commands needed to assemble the Docker image for this project.
- `README.md`: The file you are currently reading, which provides documentation for this repository.
- `app.py`: The main Python script for running the Streamlit application.
- `app_stream.py`: A Streamlit-specific version of the application script.
- `main.py`: Entry point for executing the pipeline.
- `poetry.lock` & `pyproject.toml`: Dependency management files used by Poetry.
- `schema.yaml`: YAML file defining the data schema for validation.
- `setup.py`: Setup script for installing the project as a package.
- `test.py`: Contains the test cases for the project's codebase.

## Getting Started
Running Locally using Docker:
 - Build the Docker image: 
 <pre><code> ```shell
 docker build -t sysforecast
 ```
 </code> </pre>
 - Run the Docker container: 
 <pre><code> ```shell
 docker run -p 8501:8501 sysforecast
 ```
 </code> </pre>
 - Running Locally on Streamlit:
<pre><code> ```shell
 streamlit run app_stream.py
```
</code> </pre>

## Contribution
Your contributions are always welcome! Here's how you can contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b my-new-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin my-new-feature`).
5. Create a new Pull Request.

Please make sure to update tests as appropriate.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


## Appendix I
### Hyperparameter Tunning

#### objective:
The choice depends on the type of problem:
For regression tasks: 'reg:squarederror', 'reg:pseudohubererror'.
For binary classification: 'binary:logistic', 'binary:hinge'.
For multiclass classification: 'multi:softmax', 'multi:softprob'.

#### colsample_bytree:
Typical values range from 0.3 to 0.8.
This parameter specifies the fraction of features (columns) to use for each tree. A higher value means more features will be considered, potentially increasing the model's complexity and risk of overfitting.

#### max_depth:
Common values are between 3 and 10.
This parameter controls the maximum depth of each tree. Deeper trees can model more complex patterns but might lead to overfitting. Shallower trees are faster to train and can help in preventing overfitting but might underfit.

#### alpha (L1 regularization term):
Typical values are 0, 0.001, 0.01, 0.1, 1, 10.
Regularization can help prevent overfitting. A value of 0 means no regularization. Higher values mean stronger regularization, potentially leading to simpler models (but too high a value might cause underfitting).

#### n_estimators (number of trees):
Common values range from 100 to 1000.
This determines the number of trees in the ensemble. More trees can increase model complexity and predictive power, but the gains might plateau after a certain number, and too many trees can increase the risk of overfitting.

#### learning_rate:
Typical values for the learning rate in XGBoost range from 0.01 to 0.3.
The learning rate, sometimes referred to as eta in XGBoost, controls the step size at each iteration while moving toward a minimum of the loss function. A smaller learning rate requires more boosting rounds (trees) to achieve the same reduction in residual error as a larger learning rate, but can often lead to a more accurate model.