from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from data_it_entp.pipeline.stage_06_prediction import PredictionPipeline
from data_it_entp.components.model_evaluation import ModelEvaluation




app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST']) # route to show the predictions in a web UI
def predict():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            incresed_load =float(request.form['increased_load'])
            df = pd.read_csv('artifacts/data_ingestion/allmetrics.csv')      
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
            return render_template('results.html', prediction = str(impacted_servers))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8081)