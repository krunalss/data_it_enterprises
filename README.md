# SysForecast: What-If Analysis Tool for IT Enterprises

## Live Application
The application is deployed and can be accessed on Streamlit Cloud. Experience the live application here:
[Streamlit App](https://dataitenterprises-whatif.streamlit.app/)
Also for [MlFlow experimentation](https://dagshub.com/krunalss/data_it_enterprises.mlflow/)

## Problem Statement
In the complex landscape of modern IT systems, understanding and predicting the outcomes of changes in the infrastructure poses a significant challenge. The interdependencies among various components make it difficult to anticipate the effects of workload fluctuations or modifications, often leading to inefficiency and high operational costs. SysForecast aims to address this issue by offering a 'what-if' analysis tool, enabling system administrators to model potential changes and assess their impact on system performance and resource utilization. This tool is designed to facilitate informed decision-making and reduce risks in dynamic IT environments.

## Approach
SysForecast integrates advanced machine learning algorithms with Elasticsearch to predict the impact of hypothetical changes in IT systems. The project is structured into a modular pipeline, ensuring scalability and maintainability. We leverage the robustness of Elasticsearch for efficient data ingestion and use XGBoost and SVM (Support Vector Machines) for predictive modeling. This combination allows for a comprehensive what-if analysis, providing insights into potential system alterations.

## Methodology
The SysForecast project follows a sequential pipeline comprising the following stages:

1. **Data Ingestion:** 
The Data Ingestion pipeline in this code efficiently handles data acquisition and preparation. It automates the downloading of data from a specified API, checking for pre-existing files to avoid redundancy, and then seamlessly extracts the contents of zip files into a designated directory, preparing the data for subsequent processing stages.

2. **Data Validation:** Ensuring the quality and consistency of ingested data against predefined schemas, which is crucial for reliable predictions.

3. **Data Transformation:** Cleaning and converting raw data into a format suitable for machine learning algorithms.

4. **Model Training:** Training two machine learning models—SVM and XGBoost. SVM excels in high-dimensional spaces, while XGBoost is a gradient boosting framework known for its performance in various competitions.

5. **Model Evaluation:** Evaluating model performance using metrics such as accuracy, precision, recall, and F1-score to ensure reliability in predictions.

Paramters for XGBoost
colsample_bytree: The subsample ratio of columns when constructing each tree.
learning_rate: Step size shrinkage used to prevent overfitting.
max_depth: Maximum depth of a tree.
alpha: L1 regularization term on weights.
n_estimators: Number of trees you want to build.

6. **Prediction:** Using the trained models to predict the outcomes of various what-if scenarios, assisting system administrators in making informed decisions.

Each component of the pipeline is built within a Python-based framework, ensuring a seamless flow from data ingestion to prediction.

## Streamlit Application
To enhance user interaction, a Streamlit application is developed, enabling users to easily interact with the model and visualize the potential impact of changes in IT infrastructure.

## Repository Structure
Below is an outline of the key directories and files in this repository, and their purpose:

- `artifacts/`: Contains the output files and data generated by the pipeline, including models, figures, and reports.
- `config/`: Houses configuration files used throughout the project to maintain consistency in settings and parameters.
- `data_it_entp/`: Core directory for the project's Python modules, including:
  - `components/`: Contains the individual Python scripts for each stage of the pipeline:
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
1. To Run locally on falsk, run app.py 
    python app.py
2. To Run locally on Streamlit,
    streamlit run app_stream.py  

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

### Appendix I
Hyperparameter Tunning

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