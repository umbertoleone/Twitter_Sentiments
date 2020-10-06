
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
from model import tweet_predict
import model

#creating instance of the class
app=Flask(__name__)

#run the model while starting flask
# comment the below line after the first run
# model.model_start()


def ValuePredictor(tweet):
    result = model.tweet_predict(tweet)
    return result

#to tell flask what url shoud trigger the function index()
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/visuals.html')
def about():
    return flask.render_template('visuals.html')


@app.route('/result.html',methods = ['POST'])
def result():
    if request.method == 'POST':
        print(request.form)
        to_predict_list = request.form.to_dict()
        print (to_predict_list)
        result = ValuePredictor(to_predict_list)
        return render_template("result.html", prediction=result)

    return 
    
if __name__ == "__main__":
    app.run(debug=True)
