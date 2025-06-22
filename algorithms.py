import base64
import math, statistics
import json
from openai import OpenAI, AzureOpenAI
from requests import session
import mimetypes


def gpt_call(prompt, client: OpenAI, model="gpt-4o-mini", temperature=0.4):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content.strip()
    try:
        return content
    except:
        return 10  # fallback default


def azure_gpt_call(prompt, client: AzureOpenAI, model="aigit-gpt-4.1", temperature=0.4):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content.strip()
    try:
        score = float(content)
        return max(1, min(10, score))  # clip to 1-10
    except:
        return 10  # fallback default


def gpt_call_with_images(prompt, image_path1, image_path2, client: OpenAI, model="gpt-4o", temperature=0.2):
    with open(image_path1, "rb") as image_file1, open(image_path2, "rb") as image_file2:
        image_bytes1 = base64.b64encode(image_file1.read()).decode()
        image_bytes2 = base64.b64encode(image_file2.read()).decode()

        mime1 = mimetypes.guess_type(image_path1)[0] or "application/octet-stream"
        mime2 = mimetypes.guess_type(image_path2)[0] or "application/octet-stream"
        print(f"OpenAI Image 1 MIME type: {mime1}, Image 2 MIME type: {mime2}\n\n")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime1};base64,{image_bytes1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:{mime2};base64,{image_bytes2}"}}
                ]}
            ],
            temperature=temperature
        )
    try:
        raw_content = response.choices[0].message.content.strip()
        print("Raw Model Output:", raw_content)
        parsed = json.loads(raw_content)
        print("Parsed Model Output:", parsed)
        return parsed  # returns dict: {"score": float, "reason": str}
    except Exception as e:
        return {"score": 10.0, "reason": "Default fallback due to parsing error : " + str(e)}


def azure_gpt_call_with_images(
    prompt,
    image_path1,
    image_path2,
    client: AzureOpenAI,
    model="aigit-gpt-4.1",
    temperature=0.2,
):
    with open(image_path1, "rb") as image_file1, open(image_path2, "rb") as image_file2:
        image_bytes1 = base64.b64encode(image_file1.read()).decode()
        image_bytes2 = base64.b64encode(image_file2.read()).decode()


        mime1 = mimetypes.guess_type(image_path1)[0] or "application/octet-stream"
        mime2 = mimetypes.guess_type(image_path2)[0] or "application/octet-stream"
        print(f"Azure Image 1 MIME type: {mime1}, Image 2 MIME type: {mime2}\n\n")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime1};base64,{image_bytes1}"}},
                        {"type": "image_url", "image_url": {"url": f"data:{mime2};base64,{image_bytes2}"}}
                    ]
                },
            ],
            temperature=temperature
        )
    try:
        parsed = json.loads(response.choices[0].message.content.strip())
        return parsed  # returns dict: {"score": float, "reason": str}
    except:
        return {"score": 10.0, "reason": "Default fallback due to parsing error."}


#### Vocal Analysis Functions ####
def analyze_response_vocal(words, segments, client: OpenAI | AzureOpenAI = None):
    # Clarity
    logps = [seg.avg_logprob for seg in segments]
    if logps:
        mean_conf = sum(math.exp(lp) for lp in logps) / len(logps)
        clarity = round(mean_conf * 10, 1)
    else:
        clarity = None

    # Pacing
    start_time = words[0].start
    end_time = words[-1].end
    num_words = len(words)
    duration_secs = end_time - start_time
    total_duration = end_time - start_time
    wpm = (len(words) / total_duration) * 60 if total_duration else 0

    # WPM-based pacing score
    # Normal interview speech: ~110–150 WPM is ideal
    print(f"Words: {num_words}, Duration: {duration_secs:.2f} secs, WPM: {wpm:.2f}")

    def wpm_to_score(wpm):
        if wpm <= 60:
            return 6
        if wpm <= 120:
            return 8
        if wpm <= 200:
            return 10
        return 3  # too fast

    pacing_score = wpm_to_score(wpm)

    # Variability
    durations_per_word = [w.end - w.start for w in words]
    if len(durations_per_word) > 1:
        mu = statistics.mean(durations_per_word)
        sigma = statistics.stdev(durations_per_word)
        cv = sigma / mu if mu else 0
    else:
        cv = 0

    # Variability scoring — encourage moderate variability (0.2-0.5 range)
    def variability_to_score(cv):
        if cv < 0.1:
            return 1 
        if 0.1 <= cv <= 0.3:
            return 6 + (cv - 0.1) * 10
        if 0.3 < cv <= 0.5:
            return 8
        if 0.5 < cv <= 0.7:
            return 10
        return 5 

    variability_score = round(variability_to_score(cv), 1)

    return {
        "clarity_score": clarity,
        "pacing_score": pacing_score,
        "pacing_variability_score": variability_score,
    }


def analyze_response_content(question: str, answer: str, client: OpenAI | AzureOpenAI):
    ### 1. Relevance Score
    relevance_prompt = f"""
    You are an interview evaluation system.

    Question: "{question}"
    Candidate's Answer: "{answer}"

    Evaluate how well the answer addresses the question.

    Return your response strictly in this JSON format:

    {{
      "score": <a number from 1 to 10>,
      "reason": "<a brief 1–2 sentence justification>"
    }}

    ONLY return the JSON object. No extra text or explanation.
    """

    relevance_score = (
        gpt_call(relevance_prompt, client, model="gpt-4o-mini", temperature=0)
        if isinstance(client, OpenAI)
        else azure_gpt_call(relevance_prompt, client, model="aigit-gpt-4.1", temperature=0)
    )

    ### 2️⃣ Completeness Score
    completeness_prompt = f"""
    You are an interview evaluation system. Evaluate how complete and sufficiently detailed the following answer is, given the question.

    Question: "{question}"
    Answer: "{answer}"

    Consider whether the candidate elaborated their points and provided enough depth.

    Return your response strictly in this JSON format:

    {{
      "score": <a number from 1 to 10>,
      "reason": "<a brief 1–2 sentence justification>"
    }}

    ONLY return the JSON object. No extra text or explanation.
    """

    completeness_score = (
        gpt_call(completeness_prompt, client, model="gpt-4o-mini", temperature=0)
        if isinstance(client, OpenAI)
        else azure_gpt_call(completeness_prompt, client, model="aigit-gpt-4.1", temperature=0)
    )

    ### 3️⃣ Structure Score
    structure_prompt = f"""
    You are evaluating the structure of a candidate's response to a behavioral interview question.

    Question: "{question}"

    Answer: "{answer}"

    Rate the structural quality of the response on a scale from 1 to 10. Consider:

    - Does the answer have a clear and logical flow and does not have many filler words?
    - Is it easy to follow?
    - Does it follow any structured approach like STAR (Situation, Task, Action, Result) *or* provide a coherent narrative?

    A structured answer does **not** need to follow STAR if the question is not suited for it (e.g., "Tell me about yourself"). Good structure includes clarity, progression of ideas, and logical organization.

    Return your response strictly in this JSON format:

    {{
    "score": <a number between 1 and 10>,
    "reason": "<1-2 sentence explanation>"
    }}

    ONLY return the JSON object. Do NOT include any extra text or commentary.
    """

    structure_score = (
        gpt_call(structure_prompt, client, model="gpt-4o-mini", temperature=0)
        if isinstance(client, OpenAI)
        else azure_gpt_call(structure_prompt, client, model="aigit-gpt-4.1", temperature=0)
    )

    return {
        "relevance": relevance_score,
        "completeness": completeness_score,
        "structure": structure_score,
    }



def analyze_response_visual(image1, image2, client: OpenAI | AzureOpenAI):
    """
    Takes raw model scores (1-10) from vision analysis and returns weighted visual score.
    """

    def clamp(score):
        return max(1, min(10, score))

    camera_zone_prompt = """
    You are an interview video evaluator. Look at the 2 images.

    Evaluate how well the candidate is framed and lit for a professional interview. Consider:

    - Is the candidate's head, shoulders, and upper chest clearly visible?
    - Is the candidate centered in the frame?
    - Is the lighting good (no harsh shadows, face clearly visible)?
    - Is the background clean and not distracting?

    Give a score from 1 to 10 using this scale:
    - 1–3: Poor framing and lighting (face cut off, bad lighting, busy background)
    - 4–7: Adequate but with noticeable issues (slightly off-center, uneven lighting)
    - 8–10: Excellent professional setup (well-framed, good lighting, clear background)

    Return your response strictly in this JSON format:
    
    {
    "score": <a number between 1 and 10>,
    "reason": "<1-2 sentence reason>"
    }

    ONLY return the JSON object. Do NOT add any commentary, prefixes, or explanations.
    """

    body_language_prompt = """
    You are an interview coach analyzing a candidate's body language in these 2 images.

    Evaluate based on:
    - Eye contact with the camera
    - Confident, natural posture
    - Visible hands or subtle gestures
    - Relaxed and open presence

    Give a score from 1 to 10 using this scale:
    - 1–3: Very poor (slouched, nervous, avoids eye contact)
    - 4–7: Adequate (mild stiffness, slight nervousness)
    - 8–10: Excellent body language (confident, natural, open posture with engaging presence)

    Return your response strictly in this JSON format:
    
    {
    "score": <a number between 1 and 10>,
    "reason": "<1-2 sentence reason>"
    }

    ONLY return the JSON object. Do NOT add any commentary, prefixes, or explanations.
    """
    facial_expression_prompt = """
    You are analyzing the candidate's facial engagement during an interview using these 2 images.

    Evaluate:
    - Is the candidate smiling slightly or looking approachable?
    - Do they appear warm, natural, and engaged?
    - Are facial muscles relaxed (not tense or frozen)?
    - Does their expression match a professional interview tone?

    Return a facial engagement score from 1 to 10 using this scale:
    - 1–3: Very low engagement (expressionless, tense, or uncomfortable)
    - 4–7: Adequate but somewhat flat or nervous
    - 8–10: Excellent facial engagement (natural smile or warmth, relaxed muscles, professional demeanor)


    Return your response strictly in this JSON format:
    
    {
    "score": <a number between 1 and 10>,
    "reason": "<1-2 sentence reason>"
    }

    ONLY return the JSON object. Do NOT add any commentary, prefixes, or explanations.
    """

    camera_zone_score = (
        gpt_call_with_images(
            camera_zone_prompt,
            image1,
            image2,
            client,
            model="gpt-4o",
            temperature=0.2,
        )
        if not isinstance(client, AzureOpenAI)
        else azure_gpt_call_with_images(
            camera_zone_prompt,
            image1,
            image2,
            client,
            model="aigit-gpt-4.1",
            temperature=0.2,
        )
    )
    body_language_score = (
        gpt_call_with_images(
            body_language_prompt,
            image1,
            image2,
            client,
            model="gpt-4o",
            temperature=0.2,
        )
        if not isinstance(client, AzureOpenAI)
        else azure_gpt_call_with_images(
            body_language_prompt,
            image1,
            image2,
            client,
            model="aigit-gpt-4.1",
            temperature=0.2,
        )
    )
    facial_expression_score = (
        gpt_call_with_images(
            facial_expression_prompt,
            image1,
            image2,
            client,
            model="gpt-4o",
            temperature=0.2,
        )
        if not isinstance(client, AzureOpenAI)
        else azure_gpt_call_with_images(
            facial_expression_prompt,
            image1,
            image2,
            client,
            model="aigit-gpt-4.1",
            temperature=0.2,
        )
    )

    # Weighting: camera framing is foundational but body language is key to engagement
    # weighted_score = (
    #     0.3 * camera_zone_score
    #     + 0.5 * body_language_score
    #     + 0.2 * facial_expression_score
    # )

    # Final rounded score
    # total_visual_score = round(weighted_score, 1)

    return {
        "camera_zone": camera_zone_score,
        "body_language": body_language_score,
        "facial_expression": facial_expression_score,
    }


def vision_score(image_path1, image_path2, prompt, client: OpenAI | AzureOpenAI):
    mime1 = mimetypes.guess_type(image_path1)[0] or "application/octet-stream"
    mime2 = mimetypes.guess_type(image_path2)[0] or "application/octet-stream"
    print(f"Image 1 MIME type: {mime1}, Image 2 MIME type: {mime2}")
    with open(image_path1, "rb") as image_file1, open(image_path2, "rb") as image_file2:
        response = (
            client.chat.completions.create(
                model="gpt-4o-vision",
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:{mime1};base64,{base64.b64encode(image_file1.read()).decode()}",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:{mime2};base64,{base64.b64encode(image_file2.read()).decode()}",
                            }
                        ],
                    },
                ],
            )
            if isinstance(client, OpenAI)
            else client.chat.completions.create(
                model="aigit-gpt-4.1",
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64.b64encode(image_file1.read()).decode()}",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64.b64encode(image_file2.read()).decode()}",
                            }
                        ],
                    },
                ],
            )
        )
    score_str = response.choices[0].message.content.strip()
    try:
        score = float(score_str)
        return max(1, min(10, score))
    except:
        return 10


def generate_final_feedback(questions, responses, scores, client: OpenAI | AzureOpenAI):
    qa_blocks = []
    for idx, (q, a) in enumerate(zip(questions, responses), start=1):
        block = f"""Question {idx + 1}: {q}
    Response {idx + 1}: {a}
    """
        qa_blocks.append(block)

    qa_combined = "\n\n".join(qa_blocks)

    prompt = f"""
    You are an advanced AI interview coach. You have conducted a mock interview with a candidate.

    Here is the full interview transcript:

    {qa_combined}
    
    Internal evaluation metrics (for your reference only, do not mention these scores directly):
    - Relevance Score (content): {scores['relevance']['score']}/10 (Reason : {scores['relevance']['reason']})
    - Completeness Score (content): {scores['completeness']['score']}/10 (Reason : {scores['completeness']['reason']})
    - Structure Score (content): {scores['structure']['score']}/10 (Reason : {scores['structure']['reason']})
    - Clarity Score (speech): {scores['clarity']}/10 
    - Pacing Score (speech): {scores['pacing']}/10 
    - Pacing Score evaluation: (6 is very slow, 8 is slow, 10 is ideal pace, 3 is too fast to comprehend)
    - Pacing Variability (speech): {scores['pacing_variability']}/10
    - Pacing variability score evaluation: (1 is very monotonous, 6-8 is not much variable, 10 is ideal variability, 5 is too erratic)
    - Framing & Lighting Score (visual): {scores['camera_zone']['score']}/10 (Reason : {scores['camera_zone']['reason']})
    - Body Language Score (visual): {scores['body_language']['score']}/10 (Reason : {scores['body_language']['reason']})
    - Facial Expression Score (visual): {scores['facial_expression']['score']}/10 (Reason : {scores['facial_expression']['reason']})
    ---

    Now, based on these internal evaluations, provide a **personalized interview coaching feedback** following these rules:

    - Use second-person language ("you", "your").
    - Start with a positive compliment.
    - Include 2-3 paragraph for each section(Strength, Areas to improve, Tips)
    - Provide 2-3 clear, constructive coaching tips to help improve future answers.
    - Give examples where appropriate (e.g. STAR method, storytelling).
    - Do not mention scores or numbers or criteria.
    - End with an encouraging closing sentence to motivate continued improvement.
    - Keep the tone warm, supportive, friendly and human.
    - Use real coaching style that feels like a human mentor.
    - Return your response strictly in valid JSON format using this structure:
        {{
            "strengths": "...",
            "areas_to_improve": "...",
            "tips": "..."
        }}
    """

    response = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,  # bit more creative tone
        )
        if isinstance(client, OpenAI)
        else client.chat.completions.create(
            model="aigit-gpt-4.1",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,  # bit more creative tone
        )
    )
    try:
        feedback = json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print("Failed to parse feedback JSON:", e)
        feedback = {"strengths": "", "areas_to_improve": "", "tips": ""}
    # --- Sub-Weights ---
    visual_weights = {
        "camera_zone": 0.5,
        "body_language": 0.3,
        "facial_expression": 0.2,
    }
    vocal_weights = {"clarity": 0.4, "pacing": 0.3, "pacing_variability": 0.3}
    content_weights = {"relevance": 0.4, "completeness": 0.3, "structure": 0.3}

    # --- Group Scores ---
    visual_score = sum(scores[key] * weight for key, weight in visual_weights.items())
    vocal_score = sum(scores[key] * weight for key, weight in vocal_weights.items())
    content_score = sum(scores[key] * weight for key, weight in content_weights.items())

    # --- Top-level Weights ---
    final_score = visual_score * 0.2 + vocal_score * 0.3 + content_score * 0.5
    print(
        f"Visual Score: {visual_score}, Vocal Score: {vocal_score}, Content Score: {content_score}"
        f"\nFinal Score (before scaling): {final_score}"
    )
    final_score = round(final_score * 100)  # Scale to 0-100
    final_score = max(1, min(100, final_score))  # Ensure it's within 1-100 range

    print(f"Final Score (after scaling): {final_score}")
    return {"feedback": feedback, "final_score": final_score}
