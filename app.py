from flask import Flask,jsonify,request,render_template,url_for
import numpy as np
import pandas as pd
import json
import pickle 

__model=None
__standard=None
__minmax=None
__col=None

app=Flask(__name__)
__standard=pickle.load(open('./ml_model/StdScaler.pkl','rb'))
__minmax=pickle.load(open('./ml_model/MMScaler.pkl','rb'))
__model = pickle.load(open('./ml_model/log_reg_mod.pkl', 'rb'))
__col=["gender", "self_employed", "credit_history", "married", "dependents", "education", "applicantincome", "coapplicantincome", "loanamount", "loan_amount_term", "property_area_rural", "property_area_semiurban", "property_area_urban"]

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/get_result',methods=['GET','POST'])
def get_result():
    loanammount=float(request.form.get('LoanAmmount'))
    loan_ammount_term=float(request.form.get('LoanAmmountTerm'))
    applicantincome=float(request.form.get('appincome'))
    coapplicantincome=float(request.form.get('coappincome'))
    education=int(request.form.get('uiGraduationStatus'))
    gender=int(request.form.get('uiGender'))
    self_employed=int(request.form.get('uiEmploy'))
    credit_history=int(request.form.get('uiCred'))
    married=int(request.form.get('uiMarr'))
    dependents=int(request.form.get('uiDep'))
    property_area=int(request.form.get('uiProp'))

    x=np.zeros(13)
    x[0]=gender
    x[1]=self_employed
    x[2]=credit_history
    x[3]=married
    x[4]=dependents
    x[5]=education
    x[6]=applicantincome
    x[7]=coapplicantincome
    x[8]=loanammount
    x[9]=loan_ammount_term
    if property_area>=0:
        x[property_area]=1
    
    x=x.reshape(1,13)
    df=pd.DataFrame(x)
    df.columns=__col
    df[['applicantincome','coapplicantincome','loan_amount_term']]=__minmax.transform(df[['applicantincome','coapplicantincome','loan_amount_term']])
    df[['loanamount']]=__standard.transform(df[['loanamount']])
    res=__model.predict(df)
    res=res[0]

    if(res==0):
        resultpy="Sorry, this loan can't be processed"
    else:
        resultpy="Congratulations!, Your loan can be processed"

    return render_template('result.html',result=resultpy)

if __name__ == '__main__':
    # load_saved_ml_model()
    app.run(debug=True)