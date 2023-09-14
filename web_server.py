import joblib
import time
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


stores_departments = {
    '1': ['Dept 1', 'Dept 2', 'Dept 3'],
    '2': ['Dept 4', 'Dept 5', 'Dept 6'],
    # Add more stores and their respective departments here
}

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

    # Get user inputs from the HTML form

    # Store
    Store = request.form.get('Store')
    if Store == None or Store == '':
        Store = 10
    else:
        Store = float(Store)
        df = pd.read_csv('.\static\stores.csv')
        Type = df[df['Store']==Store]['Type'].values[0]
        Size = df[df['Store']==Store]['Size'].values[0]

    # Dept
    Dept = request.form.get('Dept')
    if Dept == None or Dept == '':
        Dept = 10
    else:
        Dept = float(Dept)

    # Year
    Year = float(request.form.get('Year'))

    # Month
    Month = float(request.form.get('Month'))


    # IsHoliday
    IsHoliday = request.form.get('IsHoliday')
    if IsHoliday == 'Yes':
        IsHoliday = float(1)
    else:
         IsHoliday = float(0)



    # Temperature
    Temperature = request.form.get('Temperature')
    if Temperature == None or Temperature == '':
        Temperature = 10
    else:
        Temperature = float(Temperature)

    # Fuel_Price
    Fuel_Price = request.form.get('Fuel_Price')
    if Fuel_Price == None or Fuel_Price == '':
        Fuel_Price = 10
    else:
        Fuel_Price = float(Fuel_Price)

    # MarkDown1
    MarkDown1 = request.form.get('MarkDown1')
    if MarkDown1 == None or MarkDown1 == '':
        MarkDown1 = 10
    else:
        MarkDown1 = float(MarkDown1)

    # MarkDown2
    MarkDown2 = request.form.get('MarkDown2')
    if MarkDown2 == None or MarkDown2 == '':
        MarkDown2 = 10
    else:
        MarkDown2 = float(MarkDown2)

    # MarkDown3
    MarkDown3 = request.form.get('MarkDown3')
    if MarkDown3 == None or MarkDown3 == '':
        MarkDown3 = 10
    else:
        MarkDown3 = float(MarkDown3)

    # MarkDown4
    MarkDown4 = request.form.get('MarkDown4')
    if MarkDown4 == None or MarkDown4 == '':
        MarkDown4 = 10
    else:
        MarkDown4 = float(MarkDown4)

    # MarkDown5
    MarkDown5 = request.form.get('MarkDown5')
    if MarkDown5 == None or MarkDown5 == '':
        MarkDown5 = 10
    else:
        MarkDown5 = float(MarkDown5)

    # CPI
    CPI = request.form.get('CPI')
    if CPI == None or CPI == '':
        CPI = 10
    else:
        CPI = float(CPI)

    # Unemployment
    Unemployment = request.form.get('Unemployment')
    if Unemployment == None or Unemployment == '':
        Unemployment = 10
    else:
        Unemployment = float(Unemployment)


    x1 = [[Store,Dept,Year,Month,IsHoliday,Type,Size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment]]

    # Replace this with your model loading and prediction code
    result = model_function(x1)  # Replace with your actual model function
    result = f"${result:,.0f}"
    # Render the template with the result
    return render_template('index.html', result=result, Store=Store,Dept=Dept,Year=Year,Month=Month,IsHoliday=IsHoliday,Temperature=Temperature,Fuel_Price=Fuel_Price,MarkDown1=MarkDown1,MarkDown2=MarkDown2,MarkDown3=MarkDown3,MarkDown4=MarkDown4,MarkDown5=MarkDown5,CPI=CPI,Unemployment=Unemployment)

if __name__ == "__main__":
    app.run(debug=True)



