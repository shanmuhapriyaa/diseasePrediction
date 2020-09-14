import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from Train import train_model
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)
if not os.path.isfile('pcod1final.model'):
    train_model()

model = joblib.load('pcod1final.model')

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        age = posted_data['age']
        weight = posted_data['weight']
        height = posted_data['height']
        sugar_Level = posted_data['sugar_Level']
        bp_Level = posted_data['bp_Level']
        androgen_Level = posted_data['androgen_Level']
        sleep = posted_data['sleep']
        child_Count = posted_data['child_Count']
        gap_Mrg_Child = posted_data['gap_Mrg_Child']
        periods_long_week = posted_data['periods_long_week']
        irregular_periods = posted_data['irregular_periods']
        fast_food = posted_data['fast_food']
        loose_Weight = posted_data['loose_Weight']
        Hair_Growth = posted_data['Hair_Growth']
        dark_Patches = posted_data['dark_Patches']
        stress = posted_data['stress']
        any_Drugs = posted_data['any_Drugs']
        thyroid_problem = posted_data['thyroid_problem']
        treatment_Taken = posted_data['treatment_Taken']

        prediction = model.predict([[age,weight,height,sugar_Level,bp_Level,androgen_Level,sleep,child_Count,gap_Mrg_Child,periods_long_week,irregular_periods,fast_food,loose_Weight,Hair_Growth,dark_Patches,stress,any_Drugs,thyroid_problem,treatment_Taken]])[0]
        if prediction == 0:
            predicted_class = 'high_risk'
        elif prediction == 1:
            predicted_class = 'low_risk'
        else:
            predicted_class = 'mid_risk'

        return jsonify({
            'Prediction': predicted_class
        })

api.add_resource(MakePrediction, '/predict')
if __name__ == '__main__':
    app.run(debug=True)
