import joblib
import time
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)



def find_actual(Store,Dept,Year,Month):
    df = pd.read_csv('./static/stores_avg_actual.csv')
    
    filtered_df = df[(df['Store'] == Store) & (df['Dept'] == Dept) & (df['Year'] == Year) & (df['Month'] == Month)]
    
    if filtered_df.empty:
        return False
    
    return filtered_df['Weekly_Sales'].values[0]


def dicision_tree_model_run(x1):
    x1 = pd.DataFrame(x1, columns=['Store','Dept','Year','Month','IsHoliday','Type','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment'])
       
    loaded_model = joblib.load('.//Models//decision_tree_model.pkl')
    loaded_model_incoder = joblib.load('.//Models//incoder_model.pkl')
    x1 = loaded_model_incoder.transform(x1)
    result = loaded_model.predict(x1)[0]
    return result


def linear_tree_model_run(x1):
    x1 = pd.DataFrame(x1, columns=['Store','Dept','Year','Month','IsHoliday','Type','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment'])
       
    loaded_model = joblib.load('.//Models//linear_reg_model.pkl')
    loaded_model_incoder = joblib.load('.//Models//incoder_model_onehot.pkl')
    x1 = loaded_model_incoder.transform(x1)
    result = loaded_model.predict(x1)[0]
    return result


@app.route("/")
def main_page():
    return render_template('index.html')

@app.route("/model_run", methods=['POST'])
def model_run():
    
    Store = float(request.form.get('Store'))
    try:
        df = pd.read_csv('./static/stores.csv')
        Type = df[df['Store']==Store]['Type'].values[0]
        Size = df[df['Store']==Store]['Size'].values[0]
    except Exception as e:
        # Handle the error, e.g., by logging it or returning an error page.
        return render_template('error.html', error_message=str(e))
    

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

    result_actual = find_actual(Store,Dept,Year,Month)
    if result_actual is False:
        result_actual = 'NA'
    else:
        result_actual = f"${result_actual:,.0f}"

    result_tree = dicision_tree_model_run(x1)  
    result_tree = f"${result_tree:,.0f}"

    result_linear = linear_tree_model_run(x1)  
    result_linear = f"${result_linear:,.0f}"

    return render_template('index.html',result_actual=result_actual, result_linear=result_linear,result_tree=result_tree, Store=Store,Dept=Dept,Year=Year,Month=Month,IsHoliday=IsHoliday,Temperature=Temperature,Fuel_Price=Fuel_Price,MarkDown1=MarkDown1,MarkDown2=MarkDown2,MarkDown3=MarkDown3,MarkDown4=MarkDown4,MarkDown5=MarkDown5,CPI=CPI,Unemployment=Unemployment)

if __name__ == "__main__":
    app.run(debug=True)



