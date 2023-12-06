import argparse
import base64
import configparser
import json
import threading
import time

import pyaudio
import websocket
from websocket import ABNF


class IBMSpeechRecognizer:
    def __init__(self):
        self.chuck = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.rec = 5
        self.final_arr = []
        self.end = None
        self.map = {
            'us-east': 'gateway-wdc.watsonplatform.net',
            'us-south': 'stream.watsonplatform.net',
            'eu-gb': 'stream.watsonplatform.net',
            'eu-de': 'stream-fra.watsonplatform.net',
            'au-syd': 'gateway-syd.watsonplatform.net',
            'jp-tok': 'gateway-syd.watsonplatform.net',
        }

    @staticmethod
    def read_audio(self, websocket, timeout):
        global RATE
        p = pyaudio.PyAudio()
        RATE = int(p.get_default_input_device_info()['defaultSampleRate'])
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)

        print("* recording")
        rec = timeout or self.rec
        for i in range(0, int(self.rate / self.chunk * rec)):
            data = stream.read(self.chunk)
            websocket.send(data, ABNF.OPCODE_BINARY)
        stream.stop_stream()
        stream.close()
        print("* Stopping!")
        data = {"action": "stop"}
        websocket.send(json.dumps(data).encode('utf8'))
        time.sleep(1)
        websocket.close()
        p.terminate()

    def on_message(self, message):
        data = json.loads(message)
        if "results" in data:
            if data["results"][0]["final"]:
                self.final_arr.append(data)
                self.end = None
            else:
                self.end = data
            print(data['results'][0]['alternatives'][0]['transcript'])

    def on_error(self, error):
        print(error)

    def on_close(self, ws):
        if self.end:
            self.final_arr.append(self.end)
        transcript = "".join([x['results'][0]['alternatives'][0]['transcript']
                              for x in self.final_arr])
        print(transcript)

    def on_open(self, websocket):
        args = websocket.args
        data = {
            "action": "start",
            "content-type": "audio/l16;rate=%d" % RATE,
            "continuous": True,
            "interim_results": True,
            "word_confidence": True,
            "timestamps": True,
            "max_alternatives": 3
        }

        websocket.send(json.dumps(data).encode('utf8'))
        threading.Thread(target=self.read_audio,
                         args=(websocket, args.timeout)).start()

    def get_url(self):
        config = configparser.RawConfigParser()
        config.read('speech.cfg')
        region = config.get('auth', 'region')
        host = self.map[region]
        return ("wss://{}/speech-to-text/api/v1/recognize"
                "?model=en-AU_BroadbandModel").format(host)

    @staticmethod
    def get_auth(self):
        """
        Set up the authentication.
        """
        config = configparser.RawConfigParser()
        config.read('speech.cfg')
        apikey = config.get('auth', 'apikey')
        return ("apikey", apikey)

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(
            description='Transcribe Watson text in real time')
        parser.add_argument('-t', '--timeout', type=int, default=5)
        args = parser.parse_args()
        return args

    def run_everything(self):
        headers = {}
        userpass = ":".join(self.get_auth(self))
        headers["Authorization"] = "Basic " + base64.b64encode(
            userpass.encode()).decode()
        url = self.get_url()
        socket_object = websocket.WebSocketApp(url,
                                    header=headers,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        socket_object.on_open = self.on_open
        socket_object.args = self.parse_args()
        socket_object.run_forever()


def main():
    ibm = IBMSpeechRecognizer()
    ibm.run_everything()


