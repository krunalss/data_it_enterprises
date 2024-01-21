import pandas as pd
import os
from data_it_entp import logger
from sklearn.linear_model import ElasticNet
import joblib
from data_it_entp.config.configuration import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        features = ['CPU_Load', 'Memory_Usage', 'Disk_Usage']


        train_x = train_data[features]
        # print(f"train x={train_x}")
        test_x = test_data[features]
        # print(f"test x={test_x}")
        train_y = train_data[[self.config.target_column]]
        # print(train_y)
        test_y = test_data[[self.config.target_column]]
        # print(test_y)


        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))