from flask import Flask,send_from_directory, render_template,request
import audio_handler as ah
import commercial_api as ca
import ser_api as ser
import uuid
import copy
import json

app = Flask(__name__, template_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def videos(path):
    return app.send_static_file(path)

@app.route('/ad', methods=['POST'])
def ad():
    ogg = request.files['audio']
    br = request.form.get('br')
    is_commercial_enabled = request.form.get('isCommercialEnabled')

    print("bit rate: ", br)
    ar = 44100
    if br == 128000:
        ar = 48000

    ogg = ogg.read()
    result = {}
    result["models"] = ser.predict_emotions(copy.deepcopy(ogg), ar)
    if is_commercial_enabled == 'true':
        result["commercials"] = ca.emotion_detect(copy.deepcopy(ogg), str(uuid.uuid4()), 'ogg', ar)
    return json.dumps(result)
    

if __name__ == '__main__':
    #app.run(debug=False, host='0.0.0.0', threaded=True)
    app.run(ssl_context=('cert.pem', 'key.pem'), debug=True, host='0.0.0.0', threaded=True)
