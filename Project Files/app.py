from flask import Flask, render_template, request
from ibm_watsonx_ai.foundation_models import Model
import fitz  # PyMuPDF
import json

app = Flask(__name__)

# --- IBM Watsonx Credentials ---
import os

api_key = os.environ.get("IBM_API_KEY")
project_id = os.environ.get("IBM_PROJECT_ID")
base_url = os.environ.get("IBM_BASE_URL")
model_id = os.environ.get("IBM_MODEL_ID")


# --- Watsonx Query Function ---
def ask_watsonx(prompt):
    try:
        model = Model(
            model_id=model_id,
            credentials={"apikey": api_key, "url": base_url},
            project_id=project_id
        )

        result = model.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 1.0,
                "decoding_method": "sample"
            }
        )

        if isinstance(result, dict):
            return result.get("generated_text", "⚠️ No 'generated_text' in response.")
        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
                return parsed.get("generated_text", result)
            except json.JSONDecodeError:
                return result
        else:
            return "⚠️ Unexpected response type."
    except Exception as e:
        return f"❌ Watsonx Error: {str(e)}"

# --- Routes for each module ---
@app.route('/')
def home():
    return render_template("index.html", module="home")

@app.route('/requirements', methods=['GET', 'POST'])
def requirements():
    output = ""
    if request.method == 'POST':
        file = request.files.get("pdf_file")
        if file:
            text = ""
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
            prompt = f"Classify the following requirements into SDLC phases (Requirement, Design, Development, Testing, Deployment):\n{text}"
            output = ask_watsonx(prompt)
    return render_template("index.html", module="requirements", output=output)

@app.route('/codegen', methods=['GET', 'POST'])
def codegen():
    output = ""
    if request.method == 'POST':
        input_text = request.form.get("input_text")
        if input_text:
            prompt = f"Generate production-ready Python code for the following description:\n{input_text}"
            output = ask_watsonx(prompt)
    return render_template("index.html", module="codegen", output=output)

@app.route('/bugfix', methods=['GET', 'POST'])
def bugfix():
    output = ""
    if request.method == 'POST':
        input_text = request.form.get("input_text")
        if input_text:
            prompt = f"Here is some buggy code. Identify and fix the issues:\n{input_text}"
            output = ask_watsonx(prompt)
    return render_template("index.html", module="bugfix", output=output)

@app.route('/testcases', methods=['GET', 'POST'])
def testcases():
    output = ""
    if request.method == 'POST':
        input_text = request.form.get("input_text")
        if input_text:
            prompt = f"Write unit test cases (using unittest or pytest) for the following:\n{input_text}"
            output = ask_watsonx(prompt)
    return render_template("index.html", module="testcases", output=output)

@app.route('/summarizer', methods=['GET', 'POST'])
def summarizer():
    output = ""
    if request.method == 'POST':
        input_text = request.form.get("input_text")
        if input_text:
            prompt = f"Explain what the following code does:\n{input_text}"
            output = ask_watsonx(prompt)
    return render_template("index.html", module="summarizer", output=output)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    output = ""
    if request.method == 'POST':
        input_text = request.form.get("input_text")
        if input_text:
            prompt = f"User: {input_text}\nAssistant:"
            output = ask_watsonx(prompt)
    return render_template("index.html", module="chatbot", output=output)

if __name__ == '__main__':
    app.run(debug=True)


