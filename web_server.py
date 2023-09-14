import joblib
import time
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)



def model_function(x1):
    x1 = pd.DataFrame(x1, columns=['Store','Dept','Year','Month','IsHoliday','Type','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment'])
       
    loaded_model = joblib.load('.//Models//decision_tree_model.pkl')
    loaded_model_incoder = joblib.load('.//Models//incoder_model.pkl')
    x1 = loaded_model_incoder.transform(x1)
    result = loaded_model.predict(x1)[0]
    return result


@app.route("/")
def main_page():
    return render_template('index.html')

@app.route("/model_run", methods=['POST'])
def model_run():
    

    Store = request.form.get('Store')
    Store = float(Store)
    df = pd.read_csv('.\static\stores.csv')
    Type = df[df['Store']==Store]['Type'].values[0]
    Size = df[df['Store']==Store]['Size'].values[0]

    return render_template('index.html', result=5)
    
    Dept = float(request.form.get('Dept'))
    Year = float(request.form.get('Year'))
    Month = float(request.form.get('Month'))
    Temperature = float(request.form.get('Temperature'))
    Fuel_Price = float(request.form.get('Fuel_Price'))
    MarkDown1 = float(request.form.get('MarkDown1'))
    MarkDown2 = float(request.form.get('MarkDown2'))
    MarkDown3 = float(request.form.get('MarkDown3'))
    MarkDown4 = float(request.form.get('MarkDown4'))
    MarkDown5 = float(request.form.get('MarkDown5'))
    CPI = float(request.form.get('CPI'))
    Unemployment = float(request.form.get('Unemployment'))


    IsHoliday = request.form.get('IsHoliday')
    if IsHoliday == 'Yes':
        IsHoliday = float(1)
    else:
         IsHoliday = float(0)

    x1 = [[Store,Dept,Year,Month,IsHoliday,Type,Size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment]]


    result = model_function(x1)  
    result = f"${result:,.0f}"

    return render_template('index.html', result=result, Store=Store,Dept=Dept,Year=Year,Month=Month,IsHoliday=IsHoliday,Temperature=Temperature,Fuel_Price=Fuel_Price,MarkDown1=MarkDown1,MarkDown2=MarkDown2,MarkDown3=MarkDown3,MarkDown4=MarkDown4,MarkDown5=MarkDown5,CPI=CPI,Unemployment=Unemployment)

if __name__ == "__main__":
    app.run(debug=True)



