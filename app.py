import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI, AzureOpenAI
import os
from dotenv import load_dotenv
import uuid
import random
import algorithms
from statistics import mean 


load_dotenv()
app = Flask(__name__)
CORS(app)
azure_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                         azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
                         api_version="2024-12-01-preview",
                         api_key=os.getenv("AZURE_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

USE_AZURE_CLIENT = False

session_store = {}
available_voices = ["ballad", "coral", "fable", "nova", "shimmer"]
voice_name_map = {
    "nova": ["Emily", "Sarah", "Mia", "Grace", "Lily"],
    "shimmer": ["Olivia", "Chloe", "Zoe", "Hannah", "Ava"],
    "fable": ["James", "Leo", "Ethan", "Noah", "Lucas"],
    "coral": ["Sophia", "Emma", "Rachel", "Bella", "Kate"],
    "ballad": ["Alex", "Jordan", "Michael", "Graham", "Elliot", "Thomas", "Martin"],
}

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/start_session", methods=["POST"])
def start_session():
    session_id = str(uuid.uuid4())
    selected_voice = random.choice(list(voice_name_map.keys()))
    selected_name = random.choice(voice_name_map[selected_voice])

    session_store[session_id] = {
        "questions": [],
        "responses": [],
        "scores" : {
            "clarity" : [],
            "pacing" : [],
            "pacing_variability" : [],
            "relevance" : [],
            "completeness" : [],
            "structure" : [],
            "visual_frame" : [],
            "body_language" : [],
            "facial_expressions" : []
        },
        "user": {
            "name": request.form.get("name"),
            "jobDescription": request.form.get("jobDescription"),
        },
        "interviewer": {"voice": selected_voice, "name": selected_name},
    }
    
    print(f"Session started: {session_store[session_id]}")
    return jsonify({"session_id": session_id})

@app.route("/get_feedback", methods=["POST"])
def get_feedback():
    data = request.get_json()
    session_id = data.get("session_id")
    session = session_store.get(session_id)
    if not session: 
        return jsonify({"error": "Invalid session_id"}), 400
    
    def average(scores_list):
        return round(mean(scores_list), 2) if scores_list else None

    result = algorithms.generate_final_feedback(
        session["questions"],
        session["responses"],
        {
            "relevance": average(session["scores"]["relevance"]),
            "completeness": average(session["scores"]["completeness"]),
            "structure": average(session["scores"]["structure"]),
            "clarity": average(session["scores"]["clarity"]),
            "pacing": average(session["scores"]["pacing"]),
            "pacing_variability": average(session["scores"]["pacing_variability"]),
            "camera_zone": average(session["scores"]["visual_frame"]),
            "body_language": average(session["scores"]["body_language"]),
            "facial_expression": average(session["scores"]["facial_expressions"]),
        },
        client if not USE_AZURE_CLIENT else azure_client
    )

    return jsonify(result)

@app.route("/submit_response", methods=["POST"])
def submit_response():
    print("Request form: ", request.form)
    print("Request files: ", request.files)
    session_id = request.form.get("session_id")
    if session_id not in session_store:
        return jsonify({"error": "Invalid session_id"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        file_path = "temp_audio.wav"
        file.save(file_path)
        image_1 = request.files['image1']
        image_2 = request.files['image2']
        question_number = len(session_store[session_id]["questions"])
        filename1 = f"sessions/{session_id}/q{question_number}_1.jpg"
        filename2 = f"sessions/{session_id}/q{question_number}_2.jpg"

        os.makedirs(os.path.dirname(filename1), exist_ok=True)
        image_1.save(filename1)
        image_2.save(filename2)

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
        segments = whisper_response.segments
        session_store[session_id]["responses"].append(whisper_response.text)
        # Pause duration
        words = whisper_response.words
        segments = whisper_response.segments
        vocal_score = algorithms.analyze_response_vocal(words, segments, client if not USE_AZURE_CLIENT else azure_client)
        content_score = algorithms.analyze_response_content(session_store[session_id]["questions"], whisper_response.text, client if not USE_AZURE_CLIENT else azure_client)
        visual_score = algorithms.analyze_response_visual(filename1, filename2, client if not USE_AZURE_CLIENT else azure_client)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    session_store[session_id]["scores"]["clarity"].append(vocal_score["clarity_score"])
    session_store[session_id]["scores"]["pacing"].append(vocal_score["pacing_score"])
    session_store[session_id]["scores"]["pacing_variability"].append(vocal_score["pacing_variability_score"])
    session_store[session_id]["scores"]["visual_frame"].append(visual_score["camera_zone"])
    session_store[session_id]["scores"]["body_language"].append(visual_score["body_language"])
    session_store[session_id]["scores"]["facial_expressions"].append(visual_score["facial_expression"])
    session_store[session_id]["scores"]["relevance"].append(content_score["relevance"])
    session_store[session_id]["scores"]["completeness"].append(content_score["completeness"])
    session_store[session_id]["scores"]["structure"].append(content_score["structure"])

    print(f"Scores for session {session_id}: {session_store[session_id]['scores']}")
    print(f"Response submitted: {session_store[session_id]['responses'][-1]}")
    if len(session_store[session_id]["responses"]) > 6:
        return jsonify(
            {
                "type": "feedback",
                "feedback": "You have completed the interview. Great Job!",
            }
        )
    else:
        return jsonify(
            {
                "type": "transcript",
                "transcript": whisper_response.text
            }
        )


@app.route("/next_question", methods=["POST"])
def next_question():
    data = request.get_json()
    print("Parsed JSON:", data)

    if not data or "session_id" not in data:
        return jsonify({"error": "Missing or invalid session_id"}), 400
    session_id = data.get("session_id")
    print("Session ID:", session_id)
    if session_id not in session_store:
        return jsonify({"error": "Invalid session ID"}), 400

    previous_questions = session_store[session_id]["questions"]
    job_desc = session_store[session_id]["user"]["jobDescription"]
    interviewer_name = session_store[session_id]["interviewer"]["name"]
    question_count = len(previous_questions)
    user_name = session_store[session_id]["user"]["name"]
    last_user_response = session_store[session_id]["responses"][-1] if session_store[session_id]["responses"] else None
    segue = f"""Earlier, the candidate said: "{last_user_response}" 

    If it feels natural, briefly acknowledge this and build a follow-up question that flows from it. Otherwise, proceed with a fresh question.
    """ if last_user_response else ""
    if question_count >= 3:
        return jsonify({"type": "feedback"})

    if question_count == 0:
        prompt = f"""
        You're a warm, professional AI interviewer named {interviewer_name}. Start by greeting the candidate, {user_name}, in a friendly yet respectful manner. Introduce yourself briefly and naturally, and then transition into:

        "Tell me about yourself."

        Your tone should reflect enthusiasm and interest—like a good human interviewer would.

        Only return one line of natural spoken dialogue. Add ellipses (...) for slight pauses, commas where needed, and exclamation marks only when appropriate.
        """

        question = (
            azure_client.chat.completions.create(
                model=os.getenv("AZURE_DEPLOYMENT_NAME"), messages=[{"role": "system", "content": prompt}]
            ) if USE_AZURE_CLIENT else
            client.chat.completions.create(
                model="aigit-gpt-4.1", messages=[{"role": "system", "content": prompt}]
            )
        ).choices[0].message.content.strip()
    elif job_desc and 4 <= question_count <= 6:
        prompt = f"""
        You're a conversational, professional AI interviewer. Based on the following job description:

        "{job_desc}"

        Ask a relevant and **engaging** interview question without repeating any of these:
        {chr(10).join(f'- {q}' for q in previous_questions)}

        {segue}
        
        Use natural, varied phrasing. Imagine you're curious about the candidate’s experience.

        Only return one new question in a friendly, realistic tone. Add ellipses (...) for slight pauses, commas where needed, and exclamation marks only when appropriate.
        """
        question = (
            azure_client.chat.completions.create(
                model=os.getenv("AZURE_DEPLOYMENT_NAME"), messages=[{"role": "system", "content": prompt}]
            ) if USE_AZURE_CLIENT else
            client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}]
            )).choices[0].message.content.strip()
    else:
        prompt = f"""
        You're a professional interviewer who sounds conversational and human. Ask a behavioral question about teamwork, conflict resolution, leadership, or growth.

        Avoid repeating:
        {chr(10).join(f'- {q}' for q in previous_questions)}

        
        {segue}
        Use phrasing that feels natural—like you're really listening and want to know the candidate better.

        Only return one question. Add ellipses (...) for slight pauses, commas where needed, and exclamation marks only when appropriate. 
        
        Your tone should reflect enthusiasm and interest—like a good human interviewer would.
        """
        question = (
            azure_client.chat.completions.create(
                model=os.getenv("AZURE_DEPLOYMENT_NAME"), messages=[{"role": "system", "content": prompt}]
            ) if USE_AZURE_CLIENT else
            client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}]
            )).choices[0].message.content.strip()

    session_store[session_id]["questions"].append(question)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where app.py lives
    audio_dir = os.path.join(BASE_DIR, "audio", session_id)
    os.makedirs(audio_dir, exist_ok=True)
    
    audio_url = os.path.join(
        audio_dir, f"q{len(session_store[session_id]['questions'])}.mp3"
    )

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=session_store[session_id]["interviewer"]["voice"],
        input=question,
        instructions=(
            "You are a warm, enthusiastic interviewer with a friendly and engaging tone. "
            "Speak clearly and naturally, with a calm, conversational pace — not too fast. "
            "Use slight pauses and natural emphasis to sound human and curious, like you're genuinely interested in the candidate’s story. "
            "Vary your intonation to avoid sounding robotic or monotonous."
        )

    ) as response:
        response.stream_to_file(audio_url)
    with open(audio_url, "rb") as f:
        audio_data = f.read()

    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    return jsonify({
        "type": "question",
        "audio": audio_base64
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save to disk for re-use
    file_path = "temp_audio.wav"
    file.save(file_path)

    # --- Step 1: Whisper API (capture filler words and pauses) ---
    with open(file_path, "rb") as f:
        whisper_response = client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
            language="en",
            timestamp_granularities=["word", "segment"],
            prompt="Include filler words such as uh, um, like, you know, etc.",
        )
    print("Whisper Response:", whisper_response)
    # --- Step 2: GPT-4o Transcribe for confidence scoring ---
    # with open(file_path, "rb") as f:
    #     gpt4o_response = client.audio.transcriptions.create(
    #         file=f,
    #         model="gpt-4o-transcribe",
    #         response_format="json",
    #         language="en",
    #         include=["logprobs"]
    #     )

    # --- Step 3: Compute average clarity score ---
    logprobs = whisper_response.get("logprobs", [])
    print("Logprobs:", logprobs)
    clarity_score = sum(p["logprob"] for p in logprobs) / len(logprobs) if logprobs else None
    segments = [s.model_dump() for s in whisper_response.segments]
    words = [w.model_dump() for w in whisper_response.words]

    return jsonify(
        {"transcript": whisper_response.text, "segments": segments, "words": words}
    )


if __name__ == "__main__":
    app.run(debug=True)
