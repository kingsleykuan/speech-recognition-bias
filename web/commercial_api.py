import audio_handler as ah
import vokaturi as vk
import scipy.io.wavfile as wavfile
import wave
import io
import requests
import os
import base64
import json
import copy

abs_path = os.getcwd()
print(abs_path)
f = open(abs_path + '/config.json')
config = json.load(f)


API = config['API']
Key = config['Key']
vk.init(abs_path + '/' + config['Lib'])

print("Initialized... lib:", abs_path + "/" + config['Lib'])
print(API)
print(Key)
# params: audio        -- bytes : sample rate 44100hz and 2 channel
#         index        -- the index coule be a random string or an integer
#         audio_format -- 'ogg'/'wav' 
def emotion_detect(audio, index, audio_format, ar):

    # We have to deep copy for each API, in the sense that they are should not be misused
    input_vokaturi = None
    input_empath = None
    input_deep_affect = None

    if audio_format == 'ogg': 
        input_vokaturi = ah.ogg_to_2_44100_wav(io.BytesIO(audio), ar)
        input_empath = ah.ogg_to_1_11025_wav(io.BytesIO(audio), ar)
        input_deep_affect = ah.ogg_to_2_44100_wav(io.BytesIO(audio), ar)
    elif audio_format == 'wav':
        input_vokaturi = copy.deepcopy(audio)
        input_empath = ah.downSampleWav(copy.deepcopy(audio))
        input_deep_affect = ah.down_sample_wav_16000(copy.deepcopy(audio))
    else:
        print("Unknow format:{}, only ogg/was are supported".format(audio_format))
        return ""
        
    result = {'index' : index}
    
    try:
        # Vokaturi AR 44100 AC 2
        ar, vk_sample = wavfile.read(input_vokaturi)
        result['vokaturi'] = vk.detect(vk_sample, ar)
    except BaseException as error:
        result['vokaturi'] = 'No Result'
        print('Vokaturi error with message:{}'.format(error))

    try:
        # Empath AR 11025 AC 1
        response = requests.post(API['empath'], files = {"wav" : input_empath}, params = {'apikey' : Key['empath']})
        result['empath'] = json.loads(response.text)
        print('Empath: ', result['empath'])
    except BaseException as error:
        print("What is wrong: ", error.message)
        result['empath'] = 'No Result'
        print('Empath error with message:{}'.format(error))
    
    try:
        # Deep Affect AR 44100 AC 2 
        req = {'encoding':'WAV', 'sample_rate':'16000','language':'en-US', 'content': base64.b64encode(input_deep_affect.read()).decode('utf-8')}
        response = requests.post(API['deepaffect'],
        headers = {'Content-Type':'application/json'},
        data = json.dumps(req),
        params = {'apikey' : Key['deepaffect']})
        text = response.text
        print("Respones from Deep Affect:{}".format(text))
        result['deepaffect'] = json.loads(text)
    except BaseException as error:
        result['deepaffect'] = 'No Result'
        print('An exception occurred: {}'.format(error))

    return result


