#!/usr/bin/env python
# coding: utf-8


import pickle
import sklearn
from flask import Flask
from flask import request
from flask import jsonify

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

with open('model.bin', 'rb') as f:
    model = pickle.load(f)

app = Flask('predict_drug')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    X = dv.transform(patient)
    pred = model.predict(X).astype(int)[0]
    replace_dict = {0: "DrugY",
                    1: "drugX",
                    2: "drugA",
                    3: "drugC",
                    4: "drugB"}
    pred_str = replace_dict[pred]
    answer = {"Num drug": pred_str}
    return jsonify(answer)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)