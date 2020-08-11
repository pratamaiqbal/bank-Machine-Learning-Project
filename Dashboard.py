from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home_page.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/hasil', methods = ['POST','GET'])
def hasil():
    if request.method =='POST':
        input = request.form
        cs = int(input['cs'])
        geography = str(input['geography'])
        if geography.lower() == "spain":
            geography2 = 1
        elif geography.lower() == "france":
            geography2 = 0
        elif geography.lower() == "germany":
            geography2 = 2

        gender = str(input['gender'])
        if gender.lower() == "female":
            gender2 = 0
        elif gender.lower() == "male":
            gender2 = 1

        age = int(input['age'])
        tenure = int(input['tenure'])
        balance = float(input['balance'])

        prod = str(input['prod'])
        if prod == "1":
            prod2 = 1
        elif prod == "2":
            prod2 = 2
        elif prod == "3":
            prod2 = 3
        elif prod == "4":
            prod2 = 4

        cc = str(input['cc'])
        if cc.lower() == "yes":
            cc2 = 0
        elif cc.lower() == "no":
            cc2 = 1

        am = str(input['am'])
        if am.lower() == "yes":
            am2 = 0
        elif am.lower() == "no":
            am2 = 1
        
        salary = float(input['salary'])
        pred = Model.predict([[cs,geography2,gender2,age,tenure,balance,prod2,cc2,am2,salary]])[0]
        if pred == 0:
            pred2 = "Stay"
        elif pred == 1:
            pred2 = "out"
        return render_template('hasil.html', data=input, prediksi=pred2)

@app.route('/data')
def data():
    df = pd.read_csv('bank.csv')


    
    sns.countplot(df['Exited'], hue=df['Gender'])
    plt.xlabel('Keluar atau Tidak (0 : Tidak, 1 : Ya)')
    plt.ylabel('Jumlah')
    plt.title('Jumlah Nasabah Yang Bertahan atau Keluar Berdasarkan Gender')

    plt.savefig('gender.png',bbox_inches="tight") 


    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())


    result = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    plt.figure(figsize=(10,8))
    sns.stripplot(df['Exited'],df['Age'],hue=df['Gender'],palette='Set1')


    plt.savefig('age.png',bbox_inches="tight") 



    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    result2 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    sns.countplot(df['Exited'], hue=df['Geography'])
    plt.xlabel('Keluar atau Tidak (0 : Tidak, 1 : Ya)')
    plt.ylabel('Jumlah')
    plt.title('Jumlah Nasabah Yang Bertahan atau Keluar Berdasarkan Negara')


    plt.savefig('Geography.png',bbox_inches="tight") 



    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())


    result3 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    plt.figure(figsize=(10,8))
    sns.stripplot(df['Exited'],df['Balance'],hue=df['Gender'],palette='Set1')


    plt.savefig('Balance.png',bbox_inches="tight") 



    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())


    result4 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    plt.figure(figsize=(15,10))
    sns.countplot(df['Exited'], hue=df['Tenure'])
    plt.ylabel('Jumlah')
    plt.title('Jumlah Nasabah Yang Bertahan atau Keluar Berdasarkan Tenure')


    plt.savefig('Tenure.png',bbox_inches="tight") 



    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())


    result5 = str(figdata_png)[2:-1]


    return render_template('datavis.html', plot=result, plot2= result2, plot3= result3, plot4= result4, plot5= result5 )

if __name__ == "__main__":
    with open('churnpredict', 'rb') as model:
        Model = pickle.load(model)
    app.run(debug=True)