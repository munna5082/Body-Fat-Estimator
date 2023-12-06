from flask import Flask, render_template, request
import pickle
import numpy as np

#Loading the model
model = pickle.load(open("bodyfatmodel.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        density = float(request.form["density"])
        abdomen = float(request.form["abdomen"])
        chest = float(request.form["chest"])
        weight = float(request.form["weight"])
        hip = float(request.form["hip"])

        arr = [[density, abdomen, chest, weight, hip]]
        prediction = model.predict(arr)[0].round(2)

        string = "Percentahe of Body Fat Estimated is : " + str(prediction)+"%"

        return render_template("show.html", string=string)
    
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)