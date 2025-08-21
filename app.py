from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file
import mysql.connector
from mysql.connector import Error
import hashlib
import os
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
from gtts import gTTS

app = Flask(__name__)
app.secret_key = 'sqrt1234'
app.config['UPLOAD_FOLDER'] = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to Hindi-supported font
FONT_PATH = os.path.join(os.path.dirname(__file__), "NotoSansDevanagari-Regular.ttf")

# ---------- Helpers ----------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def translate_text(text, target_language="hi"):
    if text.strip():
        try:
            return GoogleTranslator(source='en', target=target_language).translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
    return text

# ---------- PDF Translator ----------
def translate_pdf(input_path, target_language="hi"):
    doc = fitz.open(input_path)
    output_path = input_path.replace('.pdf', f'_translated_{target_language}.pdf')

    for page in doc:
        page_text = page.get_text("dict")
        translated_spans = []

        for block in page_text["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    orig_text = span["text"].strip()
                    if not orig_text:
                        continue

                    # Translate text
                    translated = translate_text(orig_text, target_language)

                    # Schedule redaction over original text
                    r = fitz.Rect(span["bbox"])
                    page.add_redact_annot(r, fill=(1, 1, 1))  # White box
                    translated_spans.append((r, translated, span["size"]))

        # Apply redactions
        page.apply_redactions()

        # Insert translated text
        for r, translated, size in translated_spans:
            page.insert_text(
                (r.x0, r.y0 + 7.5),
                translated,
                fontname="noto",
                fontfile=FONT_PATH,
                fontsize=size,
                color=(0, 0, 0)
            )

    doc.save(output_path)
    doc.close()
    return output_path

# ---------- Video Translator ----------
def extract_audio(video_path, audio_path="temp_audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # small/medium/large also available
    result = model.transcribe(audio_path, language="en")
    return result["text"]

def text_to_speech(text, lang, output_audio="translated_audio.mp3"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_audio)
    return output_audio

def merge_audio_video(video_path, audio_path, output_path="translated_video.mp4"):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    final = video.set_audio(new_audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

def translate_video(video_path, lang):
    audio_path = extract_audio(video_path)
    original_text = transcribe_audio(audio_path)
    translated_text = translate_text(original_text, lang)
    translated_audio_path = text_to_speech(translated_text, lang, output_audio="translated_audio.mp3")
    output_video_path = video_path.replace(".mp4", f"_translated_{lang}.mp4")
    return merge_audio_video(video_path, translated_audio_path, output_path=output_video_path)

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('translate_page.html')

@app.route('/translate_page', methods=['GET', 'POST'])
def translate_page():
    if request.method == 'POST':
        upload_type = request.form.get("upload_type")
        lang = request.form.get('language', 'hi')

        # --------- Document Upload ---------
        if upload_type == "document":
            doc_file = request.files.get('document_file')
            if not doc_file or doc_file.filename == '':
                flash('No document selected', 'danger')
                return redirect(request.url)

            if allowed_file(doc_file.filename):
                filename = secure_filename(doc_file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                doc_file.save(path)

                output_path = translate_pdf(path, lang)
                return send_file(output_path, as_attachment=False)

            flash('Invalid document file type. Only PDF allowed.', 'danger')
            return redirect(request.url)

        # --------- Video Upload ---------
        elif upload_type == "video":
            video_file = request.files.get('video_file')
            if not video_file or video_file.filename == '':
                flash('No video selected', 'danger')
                return redirect(request.url)

            filename = secure_filename(video_file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(path)

            output_video_path = translate_video(path, lang)
            return send_file(output_video_path, as_attachment=True, mimetype="video/mp4")

        else:
            flash("Please select upload type (Document or Video).", "danger")
            return redirect(request.url)

    return render_template('translate_page.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/view_pdf/<filename>')
def view_pdf(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='application/pdf')
    flash('File not found', 'danger')
    return redirect(url_for('translate_page'))

# ---------- User Management ----------
def create_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="python_database"
        )
    except Error as e:
        print(f"Error: {e}")
        return None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = {
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'email': request.form['email'],
            'username': request.form['username'],
            'password': hash_password(request.form['password'])
        }

        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (first_name, last_name, email, username, password) VALUES (%s, %s, %s, %s, %s)", tuple(data.values()))
                conn.commit()
                flash('Registration successful', 'success')
                return redirect(url_for('login'))
            except Error as e:
                flash(f"Error: {e}", 'danger')
            finally:
                cursor.close()
                conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
                user = cursor.fetchone()
                if user:
                    session['logged_in'] = True
                    session['username'] = username
                    flash('Login successful', 'success')
                    return redirect(url_for('index'))
                flash('Invalid credentials', 'danger')
            except Error as e:
                flash(f"Error: {e}", 'danger')
            finally:
                cursor.close()
                conn.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
