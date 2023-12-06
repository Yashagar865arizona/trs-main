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
import threading
import time


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



    

    def handle_incoming_text(self, from_number, message_body, caller_name):
        schedule_prompt = (
    "Whenever I receive a scheduling message like a reminder, extract the information and format it as follows: \n\n"
    "\"Schedule '[Your Message]' to +123456789 at 'YYYY-MM-DD HH:MM'.\"\n\n"
    "For example, if the input is \"Remind me to say 'Hi' to +1234567890 next Thursday at 10 PM\", the output should be: \n\n"
    "\"Schedule 'Hi' to +1234567890 at 2021-12-09 22:00.\"\n\n"
    "Please ensure that all components of the date and time (year, month, day, hour, minute) are formatted as integers."
)

        prompt_messages = [
            {"role": "system", "content": schedule_prompt},
            {"role": "user", "content": message_body}
        ]
        reformatted_command =None

        # Sending the prompt to GPT
        gpt_response = self.gpt.generate_response(prompt_messages)
        
        # Assuming GPT response is in the format you expect
        reformatted_command = gpt_response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
       
        
        # Check if GPT has reformatted it into a scheduling command



        print(reformatted_command)
        if self.is_scheduling_command(reformatted_command, from_number):
            
            
            scheduled_message, date, time = self.extract_schedule_info(reformatted_command)
            # Start a new thread for sending the message later
            delay_thread = threading.Thread(target=self.delayed_send, args=(from_number, scheduled_message, date, time))
            delay_thread.start()


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

        

        return response_content
            

    

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
    

    

    def is_scheduling_command(self, message, from_number):
        
        
        # Basic regex to check if the text is likely a scheduling message
        if "Schedule" in message:
            return True
        return False
        
        

    def schedule_text(self, send_time, to_number, message):
        scheduler.add_job(self.send_text, 'date', run_date=send_time, args=[to_number, message])
    
    def reformat_scheduling_request(self, message_body):
        # Example GPT prompt to reformat the message
        prompt = f"Reformat this into a scheduling command: '{message_body}'"
        
        # Send the prompt to GPT and get the response (implement this based on your GPT setup)
        gpt_response = self.gpt.generate_response(prompt)

        # Assuming the response is in the format you expect
        reformatted_command = gpt_response.get('choices', [{}])[0].get('text', '').strip()

        return reformatted_command
    
    def delayed_send(self, to_number, message, date, scheduled_time):
        try:
            # Use regex to extract clean date and time
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date)
            time_match = re.search(r'(\d{2}:\d{2})', scheduled_time)

            if not date_match or not time_match:
                print("Invalid date or time format.")
                return

            clean_date = date_match.group(1)
            clean_time = time_match.group(1)

            datetime_str = f'{clean_date} {clean_time}'
            scheduled_time = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
        except ValueError as e:
            print(f"Error parsing date and time: {e}")
            return

        # Calculate delay in seconds
        now = datetime.datetime.now()
        delay_seconds = (scheduled_time - now).total_seconds()

        # Delay execution if the scheduled time is in the future
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Send the message after the delay
        self.send_text(to_number, message)

    def extract_schedule_info(self, text):
        try:
        # Find the starting position of the message
            start_msg_index = text.find("'") + 1
            end_msg_index = text.find("'", start_msg_index)
            message = text[start_msg_index:end_msg_index]

            # Extract date and time
            date_time_str = text.split("at")[-1].strip(" .'")
            # Splitting date and time
            date, time = date_time_str.split()

            return message, date, time
        except Exception as e:
            return "Error in parsing: " + str(e)
    
    def send_text(self, to_number, message):
        message = self.client.messages.create(body=message, from_=self.phone_number, to=to_number)
        
        return message.sid



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
twilio_auth_token = 'ee22c1ab96e9a3c01ed2aa1faf0a8284'  
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
    



@app.route("/", methods=["GET"])
def index(): # Goes under flask
    print("index")
    return "TARS phone system is running"
app.config['WTF_CSRF_ENABLED'] = False

if __name__ == "__main__":
    app.run(debug=True)





