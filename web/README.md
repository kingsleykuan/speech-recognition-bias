### Speech-Recognition-Bias Web Application
We incorprated commercial models [`Vokaturi`](https://vokaturi.com/), [`Empath`](https://vokaturi.com/), [`DeepAffects`](https://developers.deepaffects.com/)

And 6 bias recognition [`models`](https://github.com/wwongwk/speech-recognition-bias/blob/main/README.md)

#### Preview Screeshot
<kbd>![alt preview](https://github.com/wwongwk/speech-recognition-bias/blob/main/web/preview.png)<kbd>

#### Usage
1. Record your voice **up to 3** seconds by clicking the microphone button.
2. Check detection results on following table
3. Hover your cursor over the statistical image to view zoomed results.
4. Results table

| Vokaturi          | Empath           | DeepAffects           |
|-------------------|------------------|-----------------------|
| &#128545; Angry   | &#128528; Calm   | &#128545; Anger       |
| &#128530; Disgust | &#128521; Joy    | &#128528; Neutral     |
| &#128561; Fear    | &#128545; Anger  | &#128525; Excited     |
| &#128541; Happy   | &#128534; Sorrow | &#128534; Frustration |
| &#128528; Neutral | &#128513; Energy | &#128541; Happy       |
|                   |                  | &#128557; Sad         |  
#### Deployment
1. Requirement list:
 <ul>
  <li>Python</li>
  <li>Miniconda</li>
  <li>Pydub</li>
  <li>FFmpeg</li>
  <li>Requests</li>
  <li>Pytorch</li>
  <li>Flask</li>
  <li>Urllib3</li>
  <li>Urllib3</li>
</ul> 
 
2. Check your [`models`](https://github.com/wwongwk/speech-recognition-bias/blob/main/README.md) directory whether it is exists.
 
3. Clone project and into `web` folder, run:
  ```bash
  $ python server.py
  ```
  Access web application via `https://127.0.0.1:5000` as default
  
#### Configurations
You can modify the `config.json` in order to make sure the server running properly.
```json
  "API": {
    "empath": "...",
    "deepaffect": "..."
  }
```
 
If you running `server.py` on `arm64` device, you can change the lib to:
```json
"Lib": "lib/lib_arm64.so"
```
And model path:
 ```json
  "model_path": [{"acted_cnn_lstm" : "models/acted/cnn_lstm/"},
          {"acted_cnn_lstm_attention" : "models/acted/cnn_lstm_attention/"},
          {"acted_cnn_lstm_attention_multitask" : "models/acted/cnn_lstm_attention_multitask/"},
          {"observed_cnn_lstm" : "models/observed/cnn_lstm/"},
          {"observed_cnn_lstm_attention" : "models/observed/cnn_lstm_attention/"},
          {"observed_cnn_lstm_attention_multitask" : "models/observed/cnn_lstm_attention_multitask/"}
  ]
```
  
Finally, you can replace your own API keys by modifying config.json and change `.perm` files.
#### Ackownledgement 
Thanks to our **Team 1** IS4152/IS5452, NUS
