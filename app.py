from wsgiref import simple_server
from flask import Flask, request, render_template,url_for, app
from flask import Response
from flask_cors import CORS, cross_origin
from logistic_deploy import predObj
import os
import json




app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True


class ClientApi:

    def __init__(self):
        self.predObj = predObj()

@app.route("/Train", methods=['POST'])
#@app.route("/")
#def home():
    #return ClientApi()

def predictRoute():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is:     ', data)
            pred=predObj()
            res = pred.predict_log(data)

            result = clntApp.predObj.predict_log(data)
            print('result is        ',res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)

@app.route("/singlevaluepred", methods=['POST'])
def predict():
    if request.method == 'POST':
        Intercept = (request.form["Intercept"])
        occ_2 = float(request.form["occ_2"])
        occ_3 = float(request.form["occ_3"])
        occ_4 = float(request.form["occ_4"])
        occ_5 = float(request.form["occ_5"])
        occ_6 = float(request.form["occ_6"])
        occ_husb_2 = float(request.form["occ_husb_2"])
        occ_husb_3 = float(request.form["occ_husb_3"])
        occ_husb_4 = float(request.form["occ_husb_4"])
        occ_husb_5 = float(request.form["occ_husb_5"])
        occ_husb_6 = float(request.form["occ_husb_6"])
        rate_marriage = float(request.form["rate_marriage"])
        age = float(request.form["age"])
        yrs_married = float(request.form["yrs_married"])
        children = float(request.form["children"])
        religious = float(request.form["religious"])
        educ = float(request.form["educ"])


        data = np.array([[Intercept, occ_2, occ_3, occ_4, occ_5,
                            occ_6, occ_husb_2, occ_husb_3, occ_husb_4, occ_husb_5, occ_husb_6, rate_marriage
                                , age, yrs_married, children, religious, educ]])
        my_prediction = classifier.predict(data)

        return render_template(prediction=my_prediction)


if __name__ == "__main__":
    clntApp = ClientApi()
    app = app.run()
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()

