from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import psycopg2
import json
import os
import base64
import mimetypes
import tempfile
from docx import Document
from io import BytesIO
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import pytesseract
import fitz
import io

app = Flask(__name__)
CORS(app)

#API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
#HF_API_KEY = "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"

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

API_KEY = os.getenv("OCR_SPACE_API_KEY")
API_URL = "https://api.ocr.space/parse/image"

'''headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}'''

# --- PostgreSQL Connection ---
DB_URL=os.getenv("DB_URL")

def get_db_connection():
    return psycopg2.connect(DB_URL)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            history TEXT,
            answers TEXT,
            completed BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

initial_question = "What is your name of company?"

questions = [
    {"q": "Name of the Brand", "t": "text"},
    {"q": "Product Category", "t": "multiple_choice", "c": ["Food", "Clothing", "Cosmetics"]},
    {"q": "Name of the Product", "t": "text"},
    {"q": "Ingredients/Composition", "t": "paragraph"},
    {"q": "Source of raw materials", "t": "paragraph"},
    {"q": "Uniqueness of your product", "t": "paragraph"},
    {"q": "Do you have any unique or patented production methods?", "t": "paragraph"},
    {"q": "Do you add preservatives, artificial ingredients, additives, chemical dyes and agents?", "t": "multiple_choice", "c": ["Yes", "No", "Partially"]},
    {"q": "Location(s) of your corporate office", "t": "text"},
    {"q": "Location(s) of your production of this product", "t": "text"},
    {"q": "Current Certifications held by the company for the product (Organic, Fair Trade)", "t": "text"},
    {"q": "What part of the production process is handled by your company", "t": "paragraph"},
    {"q": "Can you establish that the product is organic from end to end", "t": "multiple_choice", "c": ["Yes", "No"]},
    {"q": "Can you demonstrate complete transparency of the process at your company for the product", "t": "multiple_choice", "c": ["Yes", "No"]},
    {"q": "Upload documents that support your transparency or organic claims", "t": "files", "c": ["png", "pdf", "jpeg", "docx", "txt"]},
    {"q": "Choose type of Farming you practice", "t": "multiple_choice", "c": ["Organic", "Regenerative", "PremaCulture", "FoodForest"]}
]

def get_few_shot_prompt():
    shots = "Example questions:\n"
    for item in questions:
        q = item["q"]
        a = "..." if item["t"] != "multiple_choice" else f"(Options: {', '.join(item['c'])})"
        shots += f"Q: {q}\nA: {a}\n"
    shots += "\n"
    return shots

'''def ask_model(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Error from Gemini:", e)
        return "Sorry, I couldn't generate a follow-up question."
        '''

'''def generate_next_question(history):
    few_shot = get_few_shot_prompt()
    history_text = "\n".join(history[-10:])
    input_text = (
        f"{few_shot} "
        f"You are an AI assistant conducting a structured interview to gather transparency data for a product.\n"
        f"Here is the conversation so far:\n{history_text}\n\n"
        f"Ask the next logical question that builds upon previous answers. Avoid repetition."
    )
    next_question = ask_model(input_text)
    print("AI Response:", next_question)
    return next_question'''

def get_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT history, answers, completed FROM sessions WHERE session_id = %s", (session_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        history = json.loads(row[0])
        answers = json.loads(row[1])
        completed = row[2]
    else:
        history = []
        answers = []
        completed = False
        save_session(session_id, history, answers, completed)

    return history, answers, completed

def save_session(session_id, history, answers, completed):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sessions (session_id, history, answers, completed)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (session_id)
        DO UPDATE SET history = EXCLUDED.history, answers = EXCLUDED.answers, completed = EXCLUDED.completed
    ''', (session_id, json.dumps(history), json.dumps(answers), completed))
    conn.commit()
    conn.close()

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def ocr_space_file(image_bytes, language="eng", filetype="JPG"):
    """
    Sends an image file (in bytes) to the OCR API and returns the extracted text.
    """
    payload = {
        "apikey": API_KEY,
        "language": language,
        "isOverlayRequired": False,
        "filetype": filetype
    }
    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            data=payload,
        )
        result = response.json()

        print("OCR API Response:", result)  

        if result.get("IsErroredOnProcessing", False):
            return {"error": result.get("ErrorMessage", "Unknown error")}

        if "ParsedResults" not in result or not result["ParsedResults"]:
            return {"error": "No text extracted or invalid response format"}

        return {"text": result["ParsedResults"][0].get("ParsedText", "")}

    except Exception as e:
        return {"error": str(e)}


def extract_text(file_path,language,use_openocr=True):
    """
    Extracts text from a given file:
    - PDF: Converts pages to images using PyMuPDF, then uses OpenOCR or Tesseract.
    - DOCX: Uses `python-docx` to extract text.
    - Images: Uses OpenOCR or Tesseract OCR.
    - TXT: Reads the plain text.
    """
    text = ""
    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap()
                img_byte_arr = io.BytesIO()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.save(img_byte_arr, format="JPEG", quality=80)
                img_bytes = img_byte_arr.getvalue()

                extracted_result = (
                    ocr_space_file(img_bytes, language, filetype="JPG") if use_openocr
                    else {"text": pytesseract.image_to_string(img)}
                )

                if "error" in extracted_result:
                    return extracted_result 

                extracted_text = extracted_result["text"]
                print(extracted_text)
                if extracted_text:
                    text += f"Page {page_num + 1}:\n{extracted_text}\n\n"

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            with open(file_path, "rb") as img_file:
                img_bytes = img_file.read()
                extracted_result = (
                    ocr_space_file(img_bytes,language,filetype="JPG") if use_openocr
                    else {"text": pytesseract.image_to_string(Image.open(file_path))}
                )

                if "error" in extracted_result:
                    return extracted_result  

                text = extracted_result["text"]

        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

    except Exception as e:
        return {"error": str(e)}

    return text.strip()

@app.route('/chats', methods=['POST'])
def chat():
    contents = []
    session_id = None
    user_response = None
    temp_file_path = None

    try:
        if request.content_type.startswith('multipart/form-data'):
            session_id = request.form.get("session_id")
            user_response = request.form.get("user_response")
            file = request.files.get("file")

            if file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_file:
                    file.save(temp_file.name)
                    temp_file_path = temp_file.name

                mime = get_mime_type(temp_file_path)

                if mime in ['application/pdf', 'image/jpeg', 'image/png', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    with open(temp_file_path, "rb") as f:
                        file_bytes = f.read()

                    extracted_result = extract_text(temp_file_path, language="eng", use_openocr=True)
                    print("Extracted text from File")
                    print(extracted_result)

                    if isinstance(extracted_result, dict) and "error" in extracted_result:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"File processing error: {extracted_result['error']}"}), 400

                    if extracted_result.strip():
                        user_response = extracted_result.strip()

                    contents.append({
                        "inline_data": {
                            "mime_type": mime,
                            "data": base64.b64encode(file_bytes).decode('utf-8')
                        }
                    })

                elif mime.startswith("audio/"):
                    try:
                        audio = AudioSegment.from_file(temp_file_path)
                        wav_path = temp_file_path + ".wav"
                        audio.export(wav_path, format="wav")

                        recognizer = sr.Recognizer()
                        with sr.AudioFile(wav_path) as source:
                            audio_data = recognizer.record(source)

                        user_response = recognizer.recognize_google(audio_data)
                        print("Transcribed text:", user_response)

                        os.unlink(wav_path)
                    except sr.UnknownValueError:
                        os.unlink(temp_file_path)
                        return jsonify({"error": "Could not understand audio."}), 400
                    except sr.RequestError as e:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"Speech recognition service error: {e}"}), 500
                    except Exception as e:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"Audio transcription failed: {str(e)}"}), 400
                else:
                    os.unlink(temp_file_path)
                    return jsonify({"error": f"Unsupported MIME type: {mime}. Supported types: PDF, JPEG, PNG, DOCX, TXT, AUDIO."}), 400

        else:
            data = request.get_json()
            session_id = data.get("session_id")
            user_response = data.get("user_response")

        history, answers, completed = get_session(session_id)

        if not history:
            history.append(initial_question)
            save_session(session_id, history, answers, completed)
            return jsonify({
                "message": initial_question,
                "completed": False,
                "question_number": 1
            })

        if user_response or contents:
            last_question = history[-1]
            if user_response:
                print(user_response)
                answers.append(user_response)
                
                history.append(f"A: {user_response}")
                save_session(session_id, history, answers, completed)
            else:
                answers.append("[File Uploaded]")
                contents.insert(0, {"text": f"Q: {last_question}\nA: [File Uploaded]"})


        if len(answers) >= 5:
            completed = True
            report_prompt = "Generate a report summarizing the following questions and answers:\n"
            for i, (q, a) in enumerate(zip(history, answers)):
                report_prompt += f"Q{i+1}: {q}\nA: {a}\n"

            contents.insert(0, {"text": report_prompt})
            response = model.generate_content(contents)
            report = response.text.strip()

            save_session(session_id, history, answers, completed)
            return jsonify({
                "message": "Thank you! All questions answered.",
                "completed": True,
                "answers": answers,
                "report": report
            })

        joined_history = "\n".join(history)
        few_shot = get_few_shot_prompt()

        prompt = f"You are an AI assistant conducting a structured interview to gather transparency data for a product.\n Here is the conversation so far:\n{joined_history}\n\nAsk the next logical question that builds upon previous answers. Avoid repetition."
        contents.insert(0, {"text": f"{few_shot} + {prompt}\nWhat is the next appropriate question to ask?"})

        model_response = model.generate_content(contents)
        next_question = model_response.text.strip()

        history.append(next_question)
        save_session(session_id, history, answers, completed)

        return jsonify({
            "message": next_question,
            "completed": False,
            "question_number": len(answers) + 1
        })

    except Exception as e:
        print("Error in /chats:", e)
        return jsonify({"error": str(e)}), 500

    finally:
      if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file: {e}")


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
