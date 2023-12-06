import os
import numpy as np
import requests
import cv2
import base64

import setuptools
import speech_recognition as sr
from pyscreenshot import grab
import io
from pathlib import Path
import threading
import mss
import sys
import openai
import elevenlabs
from typing import Dict, List, Tuple
import faiss
import pickle
import json
import face_recognition
import concurrent.futures
import soundfile as sf
import sounddevice as sd
import setuptools
import argparse
import base64
import configparser
import json
import threading
import time
import IBMSpeechRecognizer
import pyaudio

# Set environment variable to suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame

pygame.init()

# print("Waking up TARS...")

api_key = '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'
elevenlabs.set_api_key("")
elevenlabs_voice_id = 'bfdf947c36d2b7c1a8bc08cd301c1f8c'  # female voice: 21m00Tcm4TlvDq8ikWAM


class ContinuousSpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.speaking = False

    def listen_for_audio(self) -> sr.AudioData:
        print("Listening for audio...")
        audio = None
        try:
            with sr.Microphone() as source:
                print("Please speak now:")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=3)
                text = self.recognizer.recognize_google(audio)
                return text
        except Exception as e:
            self.recpgnizer = sr.Recognizer()
            print("Encountered error in Whisper", e)
        # print("Heard",audio)
        return audio

    def ibm_listen_for_audio(self):
        print("Listening for ")










class SpeechToTextConverter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key

    def transcribe(self, audio_data: sr.AudioData) -> str:
        if audio_data is None:
            return ""
        # Save the audio data to a WAV file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.get_wav_data())

        # Read the saved WAV file for transcription
        with open("temp_audio.wav", "rb") as f:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        # Print and return the transcription
        print("You: " + response.text)
        print()
        return response.text


class ImageHandler:
    def __init__(self, camera):
        self.camera = camera

    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode image to JPEG format.")
        return base64.b64encode(buffer).decode('utf-8')

    def capture_screenshot(self) -> str:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Capture the primary monitor
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)

            # Scale the screenshot to 50% of its original size
            scale_percent = 50  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_screenshot = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            # Define the file path for the scaled screenshot
            file_path = "captured_screenshot.jpg"
            # Save the scaled screenshot as a JPG file
            cv2.imwrite(file_path, resized_screenshot)
            return file_path

    def capture_camera_image(self) -> str:
        success, image = self.camera.read()
        if success:
            # Scale the image to 50% of its original size
            scale_percent = 50  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # Define the file path for the scaled image
            file_path = "captured_image.jpg"
            # Save the scaled image as a JPG file
            cv2.imwrite(file_path, resized_image)
            return file_path
        else:
            print("Failed to capture image from camera")
            return None


class FaceRecognition:
    def __init__(self, known_faces_file='known_faces.json'):
        self.known_faces_file = known_faces_file
        self.known_faces = self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists(self.known_faces_file):
            with open(self.known_faces_file, 'r') as file:
                data = file.read()
                if not data:  # Checks if the file is empty
                    return {}
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {self.known_faces_file}. Initializing with empty data.")
                    return {}
        else:
            # print(f"{self.known_faces_file} not found. Initializing with empty data.")
            return {}

    def save_known_faces(self):
        with open(self.known_faces_file, 'w') as file:
            json.dump(self.known_faces, file)

    def update_known_faces(self, name, face_image):
        if name.lower() == "unknown":
            # print("Skipping update for 'Unknown' face.")
            return

        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            face_encoding = face_encodings[0]
            self.known_faces[name] = face_encoding.tolist()
            self.save_known_faces()
        else:
            # print(f"No faces detected in the image for {name}.")
            print()

    def label_faces_in_image(self, image):
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(list(self.known_faces.values()), face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = list(self.known_faces.keys())[first_match_index]

            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        return image


class GPT:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if api_key is None:
            raise ValueError("API_KEY is not set")

    def generate_response(self, messages: list) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = self.compose_payload(messages)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()

        if 'choices' in response and response['choices']:
            return response
        else:
            print("Unexpected response format or error:", response)
            return {}

    def compose_payload(self, messages: list) -> dict:
        return {
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": 200,
        }

    @staticmethod
    def extract_token_info(response: dict) -> dict:
        return response.get('usage', {})


class TextToSpeechConverter:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_speech_file(self, text: str, name: str) -> str:
        try:
            print("here", name)
            openai.api_key = self.api_key
            speech_file_path = Path(name)

            # Create a speech synthesis response
            response = openai.audio.speech.create(
                model="tts-1-hd",
                voice="nova",
                input=text
            )

            # Save the speech file
            response.stream_to_file(speech_file_path)

            return speech_file_path
        except Exception as e:
            print("Error,", e)

    def play_audio(self, file_path: str) -> None:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the speech file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Cleanup
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # Delete the speech file
        os.remove(file_path)


class ElevenLabsTTSConverter:
    def __init__(self, voice_id: str):
        self.voice_id = voice_id
        self.voice_settings = {
            "stability": 0.08,
            "similarity_boost": 1.0,
            "speed": 2.0,
            "pitch": 1.0,
            "volume": 2.0,
            "emotion": "neutral"
        }

    def generate_speech(self, text: str) -> bytes:
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": self.voice_settings
        }
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}'

        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.content  # Returns the audio content as bytes
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    def generate_speech_file(self, text: str) -> str:
        audio_content = self.generate_speech(text)
        if audio_content:
            file_path = "elevenlabs_speech.mp3"
            with open(file_path, 'wb') as f:
                f.write(audio_content)
            return file_path
        else:
            print("Failed to generate speech")
            return None

    def play_audio(self, file_path: str) -> None:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the speech file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Cleanup
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # Delete the speech file
        os.remove(file_path)


class UnifiedTTS:
    def __init__(self, api_key: str, tts_service='openai'):
        self.tts_service = tts_service
        if tts_service == 'openai':
            self.tts_converter = TextToSpeechConverter(api_key)
        elif tts_service == 'elevenlabs':
            self.tts_converter = ElevenLabsTTSConverter(elevenlabs_voice_id)

    def generate_speech_file(self, text: str, name: str) -> str:
        return self.tts_converter.generate_speech_file(text, name)

    def play_audio(self, file_path: str):
        self.tts_converter.play_audio(file_path)


class LTM:
    def __init__(self, api_key: str, dimension: int = 1536) -> None:
        self.api_key = api_key
        openai.api_key = self.api_key
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_to_index = {}  # Maps text to its index in the Faiss index
        self.index_to_text = []  # Maps index in Faiss to text

    def generate_embedding(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
        embedding = np.array(response.data[0].embedding).astype('float32')
        return embedding

    def add_embedding_to_index(self, text: str) -> None:
        if text not in self.text_to_index:
            embedding = self.generate_embedding(text)
            self.index.add(embedding.reshape(1, -1))
            index = len(self.index_to_text)
            self.text_to_index[text] = index
            self.index_to_text.append(text)

    def search_similar_texts(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        if query_text not in self.text_to_index:
            self.add_embedding_to_index(query_text)

        query_embedding = self.generate_embedding(query_text)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        # print([(self.index_to_text[idx], distances[0][i]) for i, idx in enumerate(indices[0])])
        return [(self.index_to_text[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

    def store_dialogue_turn(self, user_text: str, ai_text: str) -> None:
        combined_text = user_text + " " + ai_text
        self.add_embedding_to_index(combined_text)

    def save_to_disk(self, index_file, mapping_file):
        try:
            faiss.write_index(self.index, index_file)
            with open(mapping_file, 'wb') as f:
                pickle.dump(self.index_to_text, f)
            print(f"LTM data saved to {index_file} and {mapping_file}.")
        except Exception as e:
            print(f"Error saving LTM data: {e}")

    def load_from_disk(self, index_file, mapping_file):
        if os.path.exists(index_file) and os.path.exists(mapping_file):
            self.index = faiss.read_index(index_file)
            with open(mapping_file, 'rb') as f:
                self.index_to_text = pickle.load(f)
            self.text_to_index = {text: i for i, text in enumerate(self.index_to_text)}
        else:
            print(f"Index or mapping file not found. Initializing new LTM data.")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.text_to_index = {}
            self.index_to_text = []


class TARS:
    def __init__(self, api_key: str, tts_service='openai', index_file='faiss_index.idx',
                 mapping_file='text_mapping.pkl'):
        self.speech_recognizer = ContinuousSpeechRecognizer()
        #HERE - SUBSTITUTE SR and ST between these two lines.
        self.speech_to_text = SpeechToTextConverter(api_key)
        self.speaking = False
        self.interrupted = 0
        # Initialize the camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Failed to open the camera.")

        self.image_handler = ImageHandler(self.camera)
        self.face_recognition = FaceRecognition()
        self.gpt = GPT(api_key)
        self.text_to_speech = UnifiedTTS(api_key, tts_service)
        self.messages = self.load_system_message()
        self.ltm = LTM(api_key)
        self.ltm.load_from_disk(index_file, mapping_file)

        self.image_counter = 0

    def load_system_message(self):
        try:
            with open('system_message.txt', 'r') as file:
                system_message = file.read()
            return [{"role": "system", "content": system_message}]
        except FileNotFoundError:
            return []

    def process_instruction(self, instruction):
        action = instruction.get("action")
        if action == "update_known_faces":
            # Assuming latest image is stored or can be retrieved
            latest_image = self.get_latest_captured_image()
            if latest_image is not None:
                self.handle_face_update(instruction["data"], latest_image)

    def handle_face_update(self, faces_data, image):
        for face in faces_data:
            name = face["name"]
            # Update the face recognition database with the name and the latest image
            self.face_recognition.update_known_faces(name, image)

    def get_latest_captured_image(self):
        # Retrieve the latest captured image file
        camera_image_file = self.image_handler.capture_camera_image()
        if camera_image_file:
            return cv2.imread(camera_image_file)
        return None

    def decode_base64_image(self, base64_string):
        # Decode the base64 string to an OpenCV image
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def update_messages(self, role: str, content: str, image_base64: str = None):
        message_content = [{"type": "text", "text": content}]
        if image_base64:
            message_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
        self.messages.append({"role": role, "content": message_content})

        # Retrieve similar messages for the user's message and update GPT context
        if role == "user":
            similar_texts = self.ltm.search_similar_texts(content, k=5)
            similar_texts_converted = [(text, float(score)) for text, score in similar_texts]
            for text, _ in similar_texts_converted:
                # Add similar texts as system messages
                self.messages.append({"role": "system", "content": [{"type": "text", "text": text}]})

        # Check for a complete dialogue turn
        if len(self.messages) >= 2 and self.messages[-2]["role"] == "user" and self.messages[-1]["role"] == "assistant":
            user_text = self.messages[-2]["content"][0]["text"]
            ai_text = self.messages[-1]["content"][0]["text"]
            self.ltm.store_dialogue_turn(user_text, ai_text)
            self.ltm.save_to_disk('faiss_index.idx', 'text_mapping.pkl')  # Save after each turn

    def audio_processing_parallel(self):
        while self.mainThreadRunning:
            transcript = self.speech_recognizer.listen_for_audio()
            # transcript = self.speech_to_text.transcribe(audio_data)
            if transcript not in ["you", "thank you so much for watching!",
                                  "どういたしまして、いつでもお手伝いをさせていただきます。さらに何か質問があれば、遠慮なく聞いてください。",
                                  ""]:
                self.speaking = True
                # if self.audios!=[]:
                #     self.interrupted+=1
                # self.audios = []
                # sd.stop()
                self.transcript.append(transcript)
            else:
                self.speaking = False

    def camera_parallel(self):
        camera_image_file = self.image_handler.capture_camera_image()
        camera_image = cv2.imread(camera_image_file) if camera_image_file else None
        labeled_camera_image = self.face_recognition.label_faces_in_image(
            camera_image) if camera_image is not None else None
        self.camera_image_base64 = self.image_handler.encode_image_to_base64(
            labeled_camera_image) if labeled_camera_image is not None else None

    def screenshot_parallel(self):
        # Capture and process screenshot
        screenshot_file = self.image_handler.capture_screenshot()
        screenshot_image = cv2.imread(screenshot_file) if screenshot_file else None
        self.screenshot_base64 = self.image_handler.encode_image_to_base64(
            screenshot_image) if screenshot_image is not None else None

    def activate(self):
        self.mainThreadRunning = True
        # Perform a test capture with the camera
        ret, test_frame = self.camera.read()
        if not ret:
            print("Failed to capture test frame from the camera.")
            return

        self.transcript = [""]
        print("TARS is active.")
        self.speaking = False
        thread = threading.Thread(target=self.audio_processing_parallel, args=())
        thread.start()
        self.audios = []
        AudioThread = threading.Thread(target=self.play, args=())
        AudioThread.start()
        last_request = ""
        print(last_request)
        try:
            while True:
                # Submit tasks to the pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    task1 = executor.submit(
                        self.camera_parallel)  # threading.Thread(target=self.camera_parallel, args=())
                    task2 = executor.submit(
                        self.screenshot_parallel)  # threading.Thread(target=self.screenshot_parallel, args=())
                    concurrent.futures.wait([task1, task2])

                # Handling false positives
                # if transcript.lower() in ["you", "thank you so much for watching!"]:
                #     continue

                # Capture and process camera image
                # screenshot_base64 = self.screenshot_base64
                # camera_image_base64 = self.screenshot_base64
                # # # Update messages with the transcript and images
                # self.update_messages("user", self.transcript[-1], self.camera_image_base64)
                # if screenshot_base64:
                #     self.update_messages("user", "User's screen: .", screenshot_base64)
                # # Send the updated messages list to GPT
                if self.transcript[-1] != last_request:
                    print("S E P E R A T E   R E Q U E S T")
                    last_request = self.transcript[-1]
                    self.update_messages("user", self.transcript[-1], None)
                    response = self.gpt.generate_response(self.messages)
                    message_content = ""
                    if 'choices' in response and response['choices']:
                        gpt_response = response['choices'][0]['message']
                        message_content = gpt_response['content']
                        if 'content' in gpt_response:
                            content = gpt_response['content']
                            # Remove markdown formatting characters if present
                            content = content.replace('```json\n', '').replace('\n```', '').strip()
                            try:
                                # Attempt to parse the JSON content
                                parsed_content = json.loads(content)
                                # Process JSON response
                                if 'response' in parsed_content:
                                    inner_response = parsed_content['response']
                                    if 'message' in inner_response:
                                        message_content = inner_response['message']
                                        print("TARS: " + message_content)
                                        self.update_messages("assistant", message_content)

                                    if 'instructions' in inner_response:
                                        for instruction in inner_response['instructions']:
                                            self.process_instruction(instruction)
                            except json.JSONDecodeError:
                                # Fallback for handling plain text response
                                print("TARS: " + content)
                                print()
                                self.update_messages("assistant", content)
                        else:
                            print("No valid response content received from GPT.")
                    else:
                        print("Unexpected response format or error:", response)
                        print("Actual response received:", response)  # Debug print

                    # Convert response to speech
                    # print("message content: ",message_content)
                    if message_content != "":
                        thread1 = None
                        spl = message_content.split(".")
                        print(spl)
                        i = 0
                        while i < len(spl):
                            # print("Putting in",i)
                            try:
                                # if i+2<len(spl):
                                #     thread1 = threading.Thread(target=self.generate,args=('.'.join(spl[i:i+3]),str(i)+".mp3",))
                                #     # thread1 = threading.Thread(target=self.generate,args=(spl[i],str(i)+".mp3",))
                                #     thread1.start()
                                #     thread1.join()
                                #     i+=3
                                if i + 1 < len(spl):
                                    thread1 = threading.Thread(target=self.generate,
                                                               args=('.'.join(spl[i:i + 2]), str(i) + ".mp3",))
                                    # thread1 = threading.Thread(target=self.generate,args=(spl[i],str(i)+".mp3",))
                                    thread1.start()
                                    thread1.join()
                                    i += 2
                                elif i < len(spl):
                                    thread1 = threading.Thread(target=self.generate, args=(spl[i], str(i) + ".mp3",))
                                    # thread1 = threading.Thread(target=self.generate,args=(spl[i],str(i)+".mp3",))
                                    thread1.start()
                                    thread1.join()
                                    i += 1
                            except Exception as e:
                                print(e)
        except Exception as e:
            print("some error", e)
            self.mainThreadRunning = False

    def generate(self, text, name):
        audio = self.text_to_speech.generate_speech_file(text, name)
        # self.text_to_speech.play_audio(audio)
        data_read, fs = sf.read(audio)
        # print(name,data_read)
        self.audios.append([data_read, fs])

    def play(self):
        while True:
            # print("AUDIO LIST:",self.audios)
            # with lock:
            if len(self.audios) > 0:  # and not self.speaking
                sd.play(self.audios[0][0], self.audios[0][1])
                sd.wait()
                self.audios.pop(0)


if __name__ == "__main__":
    assistant = TARS(api_key)
    # assistant.audio_processing_parallel([False])
    assistant.activate()
