import joblib
import time
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from statsmodels.tsa.statespace.sarimax import SARIMAX

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



def SARIMA_model_run(*,DataFrame,Store,Month,Year):
    DataFrame.Date= pd.to_datetime(DataFrame.Date)
    df_forecast= DataFrame[(DataFrame.Store==Store) ]
    ts=df_forecast.groupby('Date')['Weekly_Sales'].mean()
    ts_train= ts.head(120)
    ts_test= ts.tail(23)
    order = (1, 1, 1) 
    seasonal_order = (1, 1, 1, 52)
    #print (f'ts_train shape: {ts_train.shape}, ts_test shape: {ts_test.shape}')

    sarima_model = SARIMAX(ts_train, order=order, seasonal_order=seasonal_order)
    sarima_results = sarima_model.fit()

    #print(sarima_results.summary())
    forecast_period = 85
    forecast = sarima_results.get_forecast(steps=forecast_period)
    forecasted_values=forecast.predicted_mean
    filtered=forecasted_values[(forecasted_values.index.year==Year) & (forecasted_values.index.month==Month)]
    flag = 0
    if filtered.empty:
        filtered = ts_train[(ts_train.index.year==Year) & (ts_train.index.month==Month)]
        flag = 1
    # filtered= forecasted_values.between(f'{Year}-{Month}-01',f'{Year}-{Month}-31')
    
    forecast_conf_int = forecast.conf_int()
    ts_pred=[]
    for i in ts_test.index:
        ts_pred.append(sarima_results.predict(pd.Timestamp(i)))
   
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Observed')
    plt.plot(forecast.predicted_mean, label='Forecast', color='red')
    plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink')
    plt.axvline(ts_train.tail(1).index,color='orange')
    plt.title('SARIMA Forecast for Store {}'.format(Store))
    plt.legend()
    plt.savefig('.//static//figure1.jpg')
    if flag:
        return f'${filtered.mean():,.0f} (Actual)'
    else:
        return f'${filtered.mean():,.0f} (Forecasted)'


@app.route("/")
def main_page():
    return render_template('index.html', flag = 1)

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



    SARIMA_result = 0
    result_tree = 0


    if request.form.get('modelSelect') == 'SARIMA':
        flag = 0
        df = pd.read_csv('./static/time_series.csv')
        SARIMA_result = SARIMA_model_run(DataFrame=df,Store=Store,Month=Month,Year=Year)

    else:
        flag = 1
        result_tree = dicision_tree_model_run(x1)  
        result_tree = f"${result_tree:,.0f}"

    

    return render_template('index.html',flag=flag, result_actual=result_actual, SARIMA_result=SARIMA_result,result_tree=result_tree, Store=Store,Dept=Dept,Year=Year,Month=Month,IsHoliday=IsHoliday,Temperature=Temperature,Fuel_Price=Fuel_Price,MarkDown1=MarkDown1,MarkDown2=MarkDown2,MarkDown3=MarkDown3,MarkDown4=MarkDown4,MarkDown5=MarkDown5,CPI=CPI,Unemployment=Unemployment)

if __name__ == "__main__":
    app.run(debug=True)



