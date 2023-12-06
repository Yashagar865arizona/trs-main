import os
import json
from twilio.rest import Client
from tars_vision import LTM
from tars_vision import GPT
import openai
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
import re
import datetime


class TarsTextHandler:
    def __init__(self, twilio_account_sid, twilio_auth_token, phone_number, ltm_instance, openai_api_key, index_file = 'faiss_index.idx', mapping_file='text_mapping.pkl'):
        self.client = Client(twilio_account_sid, twilio_auth_token)
        self.index_file = index_file
        self.mapping_file = mapping_file
        
        self.gpt = GPT(openai_api_key)  # Initialize GPT instance
        
        self.phone_number = phone_number
        self.ltm = ltm_instance
        self.ltm.load_from_disk(index_file, mapping_file)
        self.conversations_path = "conversations"

    def is_scheduling_command(self, message):
        # Define a regex pattern that matches your expected scheduling command format
        pattern = r"Schedule '(.+?)' to (\+\d+) at ([\d-]+\s[\d:]+)"
        return re.match(pattern, message) is not None

    def schedule_text(self, send_time, to_number, message):
        scheduler.add_job(self.send_text, 'date', run_date=send_time, args=[to_number, message])

    def send_text(self, to_number, message):
        message = self.client.messages.create(body=message, from_=self.phone_number, to=to_number)
        
        return message.sid

    def handle_incoming_text(self, from_number, message_body, caller_name):
        prompt_messages = [
            {"role": "system", "content": "if it is a schduling message just format it to 'Schdule Text : Date, Time' and you do not have to schedule anything just format the text:"},
            {"role": "user", "content": message_body}
        ]

        # Sending the prompt to GPT
        gpt_response = self.gpt.generate_response(prompt_messages)
        
        # Assuming GPT response is in the format you expect
        reformatted_command = gpt_response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()


        

        # Check if GPT has reformatted it into a scheduling command
        if self.is_scheduling_command(reformatted_command):
            # Extract scheduling details (assuming GPT reformatted it into an easy-to-parse format)
            scheduled_message, to_number, send_time = self.parse_scheduling_details(reformatted_command)
            print(scheduled_message)

            # Schedule the message
            self.schedule_text(send_time, to_number, scheduled_message)
        else:
            # If not a scheduling command, process normally
            conversation_file = self.get_or_create_conversation_file(from_number)
            response = self.generate_response(from_number, message_body, caller_name)
            self.ltm.store_dialogue_turn(message_body, response)
            self.ltm.save_to_disk(self.index_file, self.mapping_file)
            self.update_conversation_file(conversation_file, message_body, caller_name=caller_name)
            self.update_conversation_file(conversation_file, response, caller_name="TARS")
            self.send_text(to_number=from_number, message=response)
        
        

    

    def generate_response(self, phone_number, message_body, caller_name):
        # Load or create the conversation file
        conversation_file = self.get_or_create_conversation_file(phone_number)
        with open(conversation_file, 'r') as file:
            conversation_data = json.load(file)
        
        # Prepare the conversation history for GPT context
        conversation_history = conversation_data.get("conversation", [])
        gpt_messages = [{"role": "user" if line.startswith(caller_name + ":") else "system", "content": line.split(": ", 1)[1]} for line in conversation_history]
        gpt_messages.append({"role": "user", "content": message_body})

        # Incorporate LTM if the conversation is not new
        if not self.is_new_conversation(phone_number):
            similar_texts = self.ltm.search_similar_texts(message_body, k=5)
            for text, _ in similar_texts:
                gpt_messages.insert(0, {"role": "system", "content": text})

        # Generate response using GPT
        gpt_response = self.gpt.generate_response(gpt_messages)
        response_content = gpt_response.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Update conversation file with the new turn
        self.update_conversation_file(conversation_file, message_body, caller_name=caller_name)
        self.update_conversation_file(conversation_file, response_content, caller_name="TARS")

        # Store the new dialogue turn in LTM
        self.ltm.store_dialogue_turn(message_body, response_content)
        self.ltm.save_to_disk(self.index_file, self.mapping_file)

        return response_content
            

    def reformat_scheduling_request(self, message_body):
        # Example GPT prompt to reformat the message
        prompt = f"Reformat this into a scheduling command: '{message_body}'"
        
        # Send the prompt to GPT and get the response (implement this based on your GPT setup)
        gpt_response = self.gpt.generate_response(prompt)

        # Assuming the response is in the format you expect
        reformatted_command = gpt_response.get('choices', [{}])[0].get('text', '').strip()

        return reformatted_command

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
    
    def parse_scheduling_details(self, command):
        # Use the same regex pattern as in is_scheduling_command
        pattern = r"Schedule '(.+?)' to (\+\d+) at ([\d-]+\s[\d:]+)"
        match = re.match(pattern, command)

        if match:
            scheduled_message = match.group(1)
            to_number = match.group(2)
            send_time_str = match.group(3)

            # Convert send_time_str to a datetime object
            send_time = datetime.strptime(send_time_str, "%Y-%m-%d %H:%M")

            return scheduled_message, to_number, send_time

        # Return None or raise an exception if the command doesn't match the expected format
        return None



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
openai_api_key = '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'  
tars_handler = TarsTextHandler(
    twilio_account_sid=twilio_account_sid,
    twilio_auth_token=twilio_auth_token,
    phone_number=phone_number,
    ltm_instance=ltm,
    openai_api_key=openai_api_key
)




app = Flask(__name__)
scheduler = BackgroundScheduler(daemon=True)
scheduler.start()

@app.route("/sms_webhook", methods=['GET', 'POST'])
def sms_webhook():
    try:
        if request.method == "POST":
            from_number = request.values.get('From', None)
            message_body = request.values.get('Body', None)
            caller_name = "User"  # Modify as necessary

            response = tars_handler.handle_incoming_text(from_number, message_body, caller_name)

            if response is None:
                response = "No response generated"

            return response, 200
        else:
            return "Only POST method is accepted", 405
    except Exception as e:
        # Log the exception for debugging
        print(f"Error: {e}")
        return "An error occurred", 500
    

@app.route("/schedule_sms", methods=['POST'])
def schedule_sms():
    data = request.json
    send_time = data.get('send_time')  # ensure this is in the correct format
    to_number = data.get('to_number')
    message = data.get('message')

    tars_handler.schedule_text(send_time, to_number, message)
    return "Message scheduled", 200

@app.route("/", methods=["GET"])
def index(): # Goes under flask
    print("index")
    return "TARS phone system is running"
app.config['WTF_CSRF_ENABLED'] = False

if __name__ == "__main__":
    app.run(debug=True)





