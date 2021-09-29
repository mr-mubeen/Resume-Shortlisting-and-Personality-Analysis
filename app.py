from os import path
import re
from zipfile import Path
from flask import Flask, render_template, request
from matplotlib.pyplot import text
from werkzeug import *
import os.path
import nltk

app = Flask(__name__)


UPLOAD_FOLDER = './CV/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/" , methods = ['GET'])
def hello_world():
    return render_template("index.html")

results = ""
per = ""

@app.route("/", methods = ['GET' , 'POST'])
def hello_world1():
    if request.method == 'POST':
        if 'cv' not in request.files:
            return 'there is no file in form!'
        file1 = request.files['cv']
        global path
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename )
        file1.save(path)
        
       
    text1 = request.form['q1']
    text2 = request.form['q2']
    text3 = request.form['q3']
    text4 = request.form['q4']
    text5 = request.form['q5']

    global answers 
    answers =  text1 + text2 + text3 + text4 + text5

    import PP
    
    print (text1)
    print (text2)
    print (text3)
    print (text4)
    print (text5)
    print(answers)
    print(path)

    results = PP.results
    per = PP.matchPercentage

    print(str(per))
    print(results)

    # results = "a"
    # per = "2"

    
    return render_template("results.html" , res = results , p = per)



if __name__ == "__main__":
    app.run( debug=True)
