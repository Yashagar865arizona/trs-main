import wave

import pyaudio
import wave
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.websocket import RecognizeCallback, AudioSource
import speech_recognition as sr
import json
from os.path import join, dirname


class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_data(self, data):
        print(json.dumps(data, indent=2))

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))


# class PyAudioListener:
#    def __init__(self):
#        self.p = pyaudio.PyAudio()
#        self.CHUNKS = 1024
#        self.CHANNELS = 2
#        self.FREQ = 44100
#        self.sec = 3
#        self.FORMAT = pyaudio.paInt16
#
#
#
#    def listen_to_audio(self):
#        stream = self.p.open(format=self.FORMAT,
#                             channels=self.CHANNELS,
#                             rate=self.FREQ,
#                             frames_per_buffer=self.CHUNKS,
#                             input = True)


class IBMSRShort:

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.speaking = False
        self.authenticator = IAMAuthenticator('CA6nqYYeVitl4mvomK8U704oEp_NhLR4EyUt6tI_udkR')
        self.speech_to_text = SpeechToTextV1(
            authenticator=self.authenticator)
        self.speech_to_text.set_service_url(
            'https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/a060b1a1-8b30-4191-b335-095ddbf0c1ad')

    def listen_to_audio(self):
        print("Listening for audio...")
        audio = None
        try:
            with sr.Microphone() as source:
                print("Please speak now:")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=3)
                with open("temp.wav", "wb") as file:
                    file.write(audio.get_wav_data())
                return audio
        except Exception as e:
            self.recognizer = sr.Recognizer()
            print("Encountered error in Whisper", e)
        return audio

    def listen_and_recognise(self, type_of_recognizer="default"):
        audio_source = self.listen_to_audio()
        if type_of_recognizer == 'ibm':

            with open('temp.wav', 'rb') as wav_file:
                text = self.speech_to_text.recognize(
                    audio=wav_file,
                    content_type='audio/wav',
                    model='en-US_NarrowbandModel').get_result()
                text_to_return = text['results'][0]['alternatives'][0]['transcript']
                print(text_to_return)
                return text_to_return
        elif type_of_recognizer == 'default':
            text = self.recognizer.recognize_google(audio_source)
            print(text)
            return text


def main():
    ibm = IBMSRShort()
    ibm.listen_and_recognise(type_of_recognizer='ibm')


if __name__ == "__main__":
    main()
