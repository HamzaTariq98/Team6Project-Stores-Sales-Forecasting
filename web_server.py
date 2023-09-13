import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


def model_function(x1):
    columns = ['Store','Dept','Year','Month','IsHoliday','Type','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment']
    x1 = pd.DataFrame(x1, columns=columns)
    loaded_model = joblib.load('.//Models//decision_tree_model.pkl')
    loaded_model_incoder = joblib.load('.//Models//incoder_model.pkl')
    result = loaded_model.predict(loaded_model_incoder.transform(x1))[0]
    return result


@app.route("/")
def main_page():
    return render_template('index.html', result='......')

@app.route("/model_run", methods=['POST'])
def model_run():

    # Get user inputs from the HTML form

    # Store
    Store = request.form.get('Store')
    if Store == None or Store == '':
        Store = 10
    else:
        Store = float(Store)

    # Dept
    Dept = request.form.get('Dept')
    if Dept == None or Dept == '':
        Dept = 10
    else:
        Dept = float(Dept)

    # Year
    Year = request.form.get('Year')
    if Year == None or Year == '':
        Year = 10
    else:
        Year = float(Year)

    # Month
    Month = request.form.get('Month')
    if Month == None or Month == '':
        Month = 10
    else:
        Month = float(Month)

    # IsHoliday
    IsHoliday = request.form.get('IsHoliday')
    if IsHoliday == 'Yes':
        IsHoliday = 1
    else:
         IsHoliday = 0
    # Type
    Type = request.form.get('Type')
    if Type == None or Type == '':
        Type = 'A'


    # Size
    Size = request.form.get('Size')
    if Size == None or Size == '':
        Size = 10
    else:
        Size = float(Size)

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

    # Render the template with the result
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)



