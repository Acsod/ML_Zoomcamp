#!/usr/bin/env python
# coding: utf-8


import pickle
import sklearn
from flask import Flask
from flask import request
from flask import jsonify

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

app = Flask('get_credit')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform(client)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[:, 1][0].round(3)
    answer = {"get_credit": pred, 
              "probability": prob}
    return jsonify(answer)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)