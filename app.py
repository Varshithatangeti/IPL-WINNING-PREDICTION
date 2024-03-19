from flask import Flask, render_template, request,redirect,url_for

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
data = pd.read_csv("IPLDATA.csv")
le = LabelEncoder()
data["batting_team"] = le.fit_transform(data["batting_team"])
data["bowling_team"] = le.fit_transform(data["bowling_team"])
data["city"] = le.fit_transform(data["city"])

data.dropna(inplace=True)

X = data.drop("result", axis=1)
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('home.html')
@app.route('/result',methods=[ 'POST'])
def result():
    if request.method == "POST":
        a=int(request.form['a'])
        batting_team = input(request.form['batting_team'])
        bowling_team = input(request.form['bowling_team'])
        city = input(request.form['city'])
        runs_left = float(request.form['runs_left'])
        wickets_left = float(request.form['wickets_left'])
        balls_left = float(request.form['balls_left'])
        total_runs_x = float(request.form['total_runs_x'])
        cur_run_rate = float(request.form['cur_run_rate'])
        req_run_rate = float(request.form['req_run_rate'])

        e_batting_team = le.fit_transform([batting_team])[0]
        e_bowling_team = le.fit_transform([bowling_team])[0]
        e_city = le.fit_transform([city])[0]

        new_data = pd.DataFrame([[a,batting_team,bowling_team,city,runs_left,wickets_left,balls_left,total_runs_x,cur_run_rate,req_run_rate]],
                                columns=['a','batting_team','bowling_team','city,runs_left','wickets_left','balls_left','total_runs_x','cur_run_rate','req_run_rate'])
        Prediction = rf_classifier.predict(new_data)
        #return render_template('result.html', Prediction=Prediction)
        return redirect(url_for('result',Prediction=Prediction))
    @app.route('/result/<Prediction>')
    def result(Prediction):
        print(Prediction)
        return render_template('result.html',Prediction=Prediction)
if __name__ == '__main__':
    app.run(debug=True)

    
     