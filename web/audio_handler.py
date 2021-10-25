from pydub import AudioSegment

from scipy.io.wavfile import write
import io

def ogg_to_1_11025_wav(ogg, ar):
    audio = AudioSegment.from_raw(ogg, sample_width=2, frame_rate=ar, channels=2)
    return audio_segments_to_bytes(audio, '1', '11025')

def ogg_to_2_44100_wav(ogg, ar):
    audio = AudioSegment.from_file(ogg, sample_width=2, frame_rate=ar, channels=2)
    return audio_segments_to_bytes(audio, '2', '44100')

def get_numpy_array_from_ogg(ogg, ar):
    sample = AudioSegment.from_file(ogg, sample_width=2, frame_rate=ar, channels=2)
    buf = io.BytesIO()
    sample.export(buf, format='wav', parameters=['-ac', str(2), '-ar', str(44100)])
    return AudioSegment.from_raw(buf, sample_width=2, frame_rate=44100, channels=2).get_array_of_samples()


#### Private
def readWavFromPath(wav_file_path):
    as_audio = AudioSegment.from_file(wav_file_path)
    return audio_segments_to_bytes(as_audio, '2', '44100')

def audio_segments_to_bytes(as_audio, ac, ar):
    buf = io.BytesIO()
    as_audio.export("d.wav", format='wav')
    as_audio.export(buf, format='WAV', parameters=['-ac', ac, '-ar', ar])
    return buf

### Public
def downSampleWav(wav_buffer):
    as_audio= AudioSegment(wav_buffer.read())
    return audio_segments_to_bytes(as_audio, '1', '11025')

def down_sample_wav_16000(wav_buffer):
    as_audio= AudioSegment(wav_buffer.read())
    return audio_segments_to_bytes(as_audio, '1', '16000') 

