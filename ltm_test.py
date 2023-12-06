import os
import json
from twilio.rest import Client
from tars_vision import LTM
import openai
from openai import OpenAI

class TarsTextHandler:
	def __init__(self, twilio_account_sid, twilio_auth_token, phone_number, ltm_instance, openai_client, index_file = 'faiss_index.idx', mapping_file='text_mapping.pkl'):
		self.client = Client(twilio_account_sid, twilio_auth_token)
		self.index_file = index_file
		self.mapping_file = mapping_file
		
		self.phone_number = phone_number
		self.ltm = ltm_instance
		self.ltm.load_from_disk(index_file, mapping_file)
		self.conversations_path = "conversations"

	def send_text(self, to_number, message):
		message = self.client.messages.create(body=message, from_=self.phone_number, to=to_number)
		
		return message.sid

	def handle_incoming_text(self, from_number, message_body, caller_name):
		conversation_file = self.get_or_create_conversation_file(from_number)
		
		
		response = self.generate_response(from_number, message_body, caller_name)
		self.ltm.store_dialogue_turn(message_body, response)
		self.ltm.save_to_disk(self.index_file, self.mapping_file)
		self.update_conversation_file(conversation_file, message_body, caller_name=caller_name)
		self.update_conversation_file(conversation_file, response, caller_name="TARS")
		self.send_text(to_number=from_number, message=response)
		return response
		

	

	def generate_response(self, phone_number, message_body, caller_name):
		if self.is_new_conversation(phone_number):
			
			# Generate response using OpenAI for new conversation (updated for new API)
			response = openai_client.chat.completions.create(
				model="gpt-3.5-turbo",
				messages=[{"role": "user", "content": message_body}],
			)
			print("NEW COMNVO")
			return response.choices[0].message.content
		else:
			# Use LTM class to find similar texts for existing conversation
			similar_texts = self.ltm.search_similar_texts(message_body)
			if similar_texts:
				# Use the most similar text to inform response generation
				most_similar_text = similar_texts[0][0]
				
				# Generate a contextual response using OpenAI
				response = openai_client.chat.completions.create(
					model="gpt-3.5-turbo",
					messages=[{"role": "system", "content": "The following is a continuation of a previous conversation."},
							{"role": "user", "content": most_similar_text},
							{"role": "user", "content": message_body}],
				)
				print(most_similar_text, "SIMILAR TEXT FOUND")
				return response.choices[0].message.content
			else:
				# If no similar texts, default to OpenAI response
				
				response = openai_client.chat.completions.create(
					model="gpt-3.5-turbo",
					prompt=message_body,
					temperature=1,
					max_tokens=200,
					stop=[caller_name]
				)
				print("NO SIMILAR TEXT")
				return response.choices[0].message.content
			

	def get_or_create_conversation_file(self, phone_number):
		file_path = os.path.join(self.conversations_path, f"{phone_number}.json")
		if not os.path.exists(self.conversations_path):
			os.makedirs(self.conversations_path)
		if not os.path.exists(file_path):
			with open(file_path, 'w') as file:
				json.dump({"phone_number": phone_number, "conversation": []}, file, indent=4)
		return file_path

	def update_conversation_file(self, file_path, message, caller_name):
		with open(file_path, 'r+') as file:
			conversation = json.load(file)
			
			conversation["conversation"].append(caller_name + ": "+ message)
			file.seek(0)
			json.dump(conversation, file, indent=4)

	def is_new_conversation(self, phone_number):
		file_path = os.path.join(self.conversations_path, f"{phone_number}.json")
		return not os.path.exists(file_path)



import faiss
import numpy as np
import openai
import os
import pickle
from flask import Flask, request




openapi_key = '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'  
openai_client = openai.OpenAI(api_key= openapi_key)
ltm = LTM(api_key=openapi_key)


twilio_account_sid = 'AC7e64afea019cf2e9706eea56aab5d143'
twilio_auth_token = '93a108eb7ec3a21329021f0602fcacc7'  
phone_number = '8557520721'  
tars_handler = TarsTextHandler(
	twilio_account_sid=twilio_account_sid,
	twilio_auth_token=twilio_auth_token,
	phone_number=phone_number,
	ltm_instance=ltm,
	openai_client = openai_client
)




app = Flask(__name__)

@app.route("/sms_webhook", methods=['GET', 'POST'])
def sms_webhook():
	if request.method == "POST":
		from_number = request.values.get('From', None)
		message_body = request.values.get('Body', None)

		
		
		caller_name = "User"  # Replace with actual caller name
		
		
		
		response = tars_handler.handle_incoming_text(from_number, message_body, caller_name)
		
		print(f"Received message from {from_number}: {message_body}")
        
		

		return response, 200
	else:
		return "Please POST", 200

@app.route("/", methods=["GET"])
def index(): # Goes under flask
	print("index")
	return "TARS phone system is running"
app.config['WTF_CSRF_ENABLED'] = False

if __name__ == "__main__":
	app.run(debug=True)




