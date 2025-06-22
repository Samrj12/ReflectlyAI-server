import json
import requests
import algorithms
from openai import AzureOpenAI, OpenAI
import os

from dotenv import load_dotenv

load_dotenv()
azure_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                         azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
                         api_version="2024-12-01-preview",
                         api_key=os.getenv("AZURE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def request_test():
    # Start session
    response = requests.post("http://localhost:5000/start_session", data={
        "name": "Rudraksh",
        "jobDescription": "Data Scientist"
    })
    session_id = response.json()['session_id']
    print("Session ID:", session_id)

    # Fetch next question
    q_response = requests.post("http://localhost:5000/next_question", json={
        "session_id": session_id
    })
    print(q_response.json())

# https://aigtaihub4104782688.openai.azure.com
#aigit-gpt-4.1
# 7dblXJNlb5ZtMwKKIrEyjV5v4Bq3NlNQ9utP47V6O2yNMp0V8t9XJQQJ99BEACHYHv6XJ3w3AAAAACOGhTQ5

def vision_test():
   result = algorithms.analyze_response_visual("img.jpg","img.jpg", client)
   print(result)

def audio_test(file_path):
    with open(file_path, "rb") as f:
        whisper_response = client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
            language="en",
            timestamp_granularities=["word", "segment"],
            prompt="Include filler words such as uh, um, like, you know, etc.",
        )

        # Clarity score
        # Pause duration
    print(json.dumps(whisper_response.model_dump(), indent=2))
    words = whisper_response.words
    segments = whisper_response.segments
    vocal_score = algorithms.analyze_response_vocal(words, segments, client)
    content_score = algorithms.analyze_response_content("Hello RJ! It's so great to meet you today! Let'se start off by telling me a bit about yourself and why should we choose for this role?", whisper_response.text, client)
    print("Scores:")
    print("Vocal:", vocal_score)
    print("Content:", content_score)
    
if __name__ == "__main__":
    # request_test()
    # vision_test()
    audio_test("audio.mp3")
    # audio_test("audio.wav")
    # audio_test("audio.m4a")
    # audio_test("audio.ogg")
    # audio_test("audio.flac")