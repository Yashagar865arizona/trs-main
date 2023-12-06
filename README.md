# TARS

Imagine a world where your digital interactions are not just transactions, but experiences - deeply personal, continuously evolving, and remarkably intuitive. This is the world of TARS, a groundbreaking leap in human-machine interaction that transcends traditional boundaries of technology. Why do you need TARS? Because it's not just an advancement; it's a transformation of the digital experience. TARS listens, understands, and speaks in a voice that's uncannily real, creating a sense of comfort and familiarity in every interaction. It's not just a voice responding to your commands; it's an extension of your own voice, a reflection of your personality.

But TARS is more than just a sophisticated conversationalist. Its advanced image recognition capabilities act as an extension of your own vision, offering insights and assistance in real-time. Whether you're navigating a new city, sorting through old photographs, or exploring the world around you, TARS enhances these experiences by providing contextual, visual intelligence that's tailored to your perspective.

The true power of TARS lies in its ability to remember and evolve. Each interaction is a building block in a continuously developing relationship, mirroring the depth and complexity of human connections. This isn't a digital assistant that starts each conversation from scratch; TARS understands the ongoing narrative of your life, making every interaction more meaningful and personalized.

And in a world where digital fatigue is real, TARS stands out by being proactive. It reaches out with content that matters to you, initiates conversations that spark your interest, and provides insights that are aligned with your preferences. This level of proactive engagement is a game-changer, transforming TARS from a tool to a companion.

# TARS Phone System Documentation

## Table of Contents
- [Installation Instructions](#installation-instructions)
- [Usage Instructions](#usage-instructions)
- [Logging](#logging)
- [FAQ](#faq)
- [Feedback and Contribution](#feedback-and-contribution)

## Installation Instructions

### Step 1: Clone the Repository
Clone the designated repository to your local machine.

### Step 2: Install Dependencies
To install the necessary dependencies, open your terminal and run the following command:
````
pip install -r requirements.txt
````
Note: Make sure you're running a compatible version of Python for this project.

### Step 3: Configure API Keys
Obtain API keys for the following services and integrate them into the code. Below are brief explanations on what each service is used for:
- Twilio (Required for phone calls)
- OpenAI (Required for GPT-4 functionality)
- ElevenLabs (Required for text-to-speech)
- Ngrok (Required for hosting the server)

### Step 4: Download ngrok
Download and unzip the ngrok executable. Ngrok is used for creating a secure tunnel to your localhost, allowing you to expose a local server to the internet.
[Ngrok Download](https://ngrok.com/download)

### Step 5: Download ffmpeg
Download and install ffmpeg for audio processing. This is essential for handling various audio tasks in the project.
[FFmpeg Download](https://ffmpeg.org/download.html)

### Step 6: Add Audio Files (OPTIONAL)
Place MP3 files in the `audio_files` folder. These will be used for playback during call handling.

### Step 7: Customize Configuration Files (OPTIONAL)
Modify the `TARS.txt` and `voice_settings.json` files as needed. Customizing these files can allow you to change the behavior of TARS to better suit your needs.

## Usage Instructions

### Starting the Server
Run the following command to start the Flask server and ngrok tunnel:
````
python main.py
````

### Twilio Configuration
Configure your Twilio phone number to use the ngrok URL as follows:
- Voice Request URL: `<ngrok url>/incoming_call`
- Status Callback URL: `<ngrok url>/webhook`

### Making Calls
Call your configured Twilio number to interact with TARS.

## Logging
Phone calls will be logged in individual text files for each caller.

## FAQ

### Installation Issues

**Q: I'm getting import errors for some dependencies.**  
A: Make sure you have installed all the required packages listed in `requirements.txt`. You can run `pip install -r requirements.txt` to install them.

**Q: I don't have ffmpeg installed or configured properly.**  
A: Follow the ffmpeg install instructions carefully. On some systems, you may need to update your PATH environment variable to include the ffmpeg binary location.

**Q: Ngrok is not starting up properly or I'm having issues exposing my localhost.**  
A: Check that you've downloaded the ngrok executable for your platform and that it has execution permissions. You may need to run ngrok manually from the command line first to authorize and configure it.

### API Keys

**Q: My app isn't working and I'm seeing authorization errors.**  
A: Double check that your API keys for Twilio, OpenAI, and ElevenLabs are properly set in your code. The keys need to be valid for everything to function correctly.

**Q: I'm hitting API limits or being rate limited.**  
A: If you exceed usage quotas for any of the APIs, you may need to upgrade your account plan or optimize the app to reduce API calls. For OpenAI, you can tweak the `max_tokens` parameter to limit response length.

### Call Handling Issues

**Q: Calls are getting dropped or audio quality is poor.**  
A: There are a few things that could impact call stability and quality:
- Check your internet connection speed and reliability.
- Try reducing the ElevenLabs API speed parameter to improve synthesis.
- Adjust Twilio voice webhook configurations and timeouts as needed.

**Q: TARS is repeating themselves or conversation is getting stuck.**  
A: Limit the `maxLength` parameter for Record verbs to prevent excessively long recordings. Also customize the `TARS.txt` file to provide more variability in TARS' responses.

**Q: TARS stops responding or hangs up unexpectedly.**  
A: Review your logs for any errors. TARS will hang up if their response contains `{hangup}`. Make sure your Twilio phone number configuration is correct.

**Q: User recordings are not triggering a TARS response.**  
A: The app expects recordings from Twilio to come in as POST requests to `/handle_incoming_input`. Make sure this path is configured properly in Twilio.

### Audio Issues

**Q: TARS is not playing the audio files I added.**  
A: Put any custom audio files you want played under `audio_files/` folder. Supported formats are MP3, WAV, etc. Make sure the filenames match what is referenced in the code.

**Q: There are pops or clipping in the generated speech.**  
A: Try adjusting the ElevenLabs voice parameters like `stability` and `similarity_boost` to improve quality. Also try tweaking the gain applied to generated audio before playback.

### Customization

**Q: How do I change TARS' voice, personality or behavior?**  
A: The `TARS.txt` file contains the AI assistant's base knowledge. Customize this file for different personalities. The `voice_settings.json` file allows you to configure different voices.

**Q: Can I create multiple AI assistants?**  
A: Yes, the code supports switching identities by customizing the prompts sent to the text generation API. Add new identities and voices as needed.

**Q: How can I customize the conversation summarization?**  
A: The summary generation is handled by `update_summary()`. You can tweak this function and the summarizer prompt file to fit your specific needs.

---
Please open an issue on GitHub if you have any other questions! We're happy to help clarify any part of this project.


## Feedback and Contribution
We encourage users to provide feedback, open issues, or contribute to the project. If you wish to contribute, please follow the guidelines outlined in CONTRIBUTING.md.
