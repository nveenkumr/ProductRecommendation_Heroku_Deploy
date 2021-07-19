#from __future__ import division, print_function
# coding=utf-8
#import re
#import pandas as pd
#import pickle
#from sklearn.feature_extraction.text import TfidfVectorizer
from model import cal_top5_prod
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,render_template_string
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def recommend_prod():
    if request.method == 'POST':
        #Get the request data         
        print(request.data.decode('UTF-8'))
        user_name = request.data.decode('UTF-8')
        top_5_prods = cal_top5_prod(user_name)
        result = ",".join(top_5_prods.values)#(top_5_prods['ProductName'].values)
        
        return result  #render_template_string (result)
    elif request.method == 'GET':
        return render_template('index.html')
    return None


if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)

