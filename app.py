from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file, render_template_string
import mysql.connector
from mysql.connector import Error
import hashlib
import os
import re
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
from gtts import gTTS

# === NEW: OCR & audio utils ===
import cv2
import pytesseract
from pydub import AudioSegment
from pydub.utils import which

app = Flask(__name__)
app.secret_key = 'sqrt1234'
app.config['UPLOAD_FOLDER'] = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- Fonts per language for PDFs ----------
BASE_DIR = os.path.dirname(__file__)
FONT_MAP = {
    "hi": os.path.join(BASE_DIR, "NotoSansDevanagari-Regular.ttf"),
    "mr": os.path.join(BASE_DIR, "NotoSansDevanagari-Regular.ttf"),
    "gu": os.path.join(BASE_DIR, "NotoSansGujarati-Regular.ttf"),
    "ta": os.path.join(BASE_DIR, "NotoSansTamil-Regular.ttf"),
    "te": os.path.join(BASE_DIR, "NotoSansTelugu-Regular.ttf"),
    "pa": os.path.join(BASE_DIR, "NotoSansGurmukhi-Regular.ttf"),
    "bn": os.path.join(BASE_DIR, "NotoSansBengali-Regular.ttf"),
}
DEFAULT_FONT = os.path.join(BASE_DIR, "NotoSansDevanagari-Regular.ttf")

def font_for_lang(lang: str) -> str:
    return FONT_MAP.get(lang, DEFAULT_FONT)


# gTTS language codes mapped from your dropdown
TTS_LANG_MAP = {
    "hi": "hi",  # Hindi
    "gu": "gu",  # Gujarati
    "ta": "ta",  # Tamil
    "te": "te",  # Telugu
    "mr": "mr",  # Marathi
    "pa": "pa",  # Punjabi
    "bn": "bn",  # Bengali
}

# ---------- Helpers ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def translate_text(text, target_language):
    if text and text.strip():
        try:
            return GoogleTranslator(source='auto', target=target_language).translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
    return text or ""

# ---------- PDF Translator ----------
def translate_pdf(input_path, target_language="hi"):
    doc = fitz.open(input_path)
    output_path = input_path.replace('.pdf', f'_translated_{target_language}.pdf')
    fontfile = font_for_lang(target_language)

    for page in doc:
        page_text = page.get_text("dict")
        translated_spans = []

        for block in page_text.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    orig_text = (span.get("text") or "").strip()
                    if not orig_text:
                        continue
                    translated = translate_text(orig_text, target_language)
                    r = fitz.Rect(span["bbox"])
                    page.add_redact_annot(r, fill=(1, 1, 1))
                    translated_spans.append((r, translated, span.get("size", 11)))

        page.apply_redactions()

        for r, translated, size in translated_spans:
            try:
                page.insert_text(
                    (r.x0, r.y0 + 7.5),
                    translated,
                    fontfile=fontfile,
                    fontsize=float(size) if size else 11,
                    color=(0, 0, 0)
                )
            except Exception as e:
                print(f"insert_text error: {e}")

    doc.save(output_path)
    doc.close()
    return output_path

# ---------- Video: audio -> STT -> translate -> TTS -> merge ----------
def extract_audio(video_path, audio_path=None):
    if audio_path is None:
        root, _ = os.path.splitext(video_path)
        audio_path = root + "_audio.wav"
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise RuntimeError("No audio track found in video.")
    clip.audio.write_audiofile(audio_path)
    clip.close()
    return audio_path

# cache whisper model to speed up multiple requests
_whisper_model = None
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def transcribe_audio(audio_path):
    model = get_whisper_model()
    result = model.transcribe(audio_path)  # auto language detection
    return result.get("text", "").strip()

def _split_for_tts(text, max_len=200):
    # split into reasonable-sized chunks for gTTS
    sentences = re.split(r'(?<=[\.\!\?।])\s+', text.strip())
    out, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= max_len:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                out.append(buf)
            if len(s) <= max_len:
                buf = s
            else:
                for i in range(0, len(s), max_len):
                    out.append(s[i:i+max_len])
                buf = ""
    if buf:
        out.append(buf)
    return out

def tts_from_text(text, lang_code, output_audio):
    # Ensure ffmpeg availability for pydub (not mandatory but recommended)
    if which("ffmpeg") is None:
        print("Warning: ffmpeg not found in PATH. pydub may not work optimally.")

    chunks = _split_for_tts(text)
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], "tmp_tts")
    os.makedirs(temp_dir, exist_ok=True)

    # fallback Hindi if unknown code passed
    tts_lang = TTS_LANG_MAP.get(lang_code, "hi")

    combined = AudioSegment.silent(duration=150)
    for i, part in enumerate(chunks):
        part_path = os.path.join(temp_dir, f"part_{i}.mp3")
        gTTS(text=part, lang=tts_lang).save(part_path)
        combined += AudioSegment.from_file(part_path, format="mp3") + AudioSegment.silent(duration=120)

    combined.export(output_audio, format="mp3")
    return output_audio

def merge_audio_video(video_path, audio_path, output_path=None):
    if output_path is None:
        root, _ = os.path.splitext(video_path)
        output_path = f"{root}_translated.mp4"
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    final = video.set_audio(new_audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    video.close()
    new_audio.close()
    return output_path

# ---------- OCR on frames to get on-screen text ----------
def ocr_video_text(video_path, interval_sec=1.5, tesseract_lang="eng"):
    """
    Extract on-screen text by sampling frames every interval_sec seconds.
    Returns a deduplicated concatenated string.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps * interval_sec), 1)
    frame_idx = 0
    texts = []
    seen = set()

    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % step == 0:
            ret2, frame = cap.retrieve()
            if not ret2:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # light threshold/denoise
            gray = cv2.bilateralFilter(gray, 5, 75, 75)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            txt = pytesseract.image_to_string(thresh, lang=tesseract_lang)
            txt = re.sub(r'\s+', ' ', txt).strip()
            if txt and txt not in seen:
                texts.append(txt)
                seen.add(txt)
        frame_idx += 1

    cap.release()
    return " ".join(texts)

# ---------- Simple extractive summarizer ----------
STOPWORDS = set("""
a an the and or but if while with without within about above below over under again further then once
here there when where why how all any both each few more most other some such no nor not only own same
so than too very can will just don should now is am are was were be been being do does did doing have has had
having in into on at by for of to from as that this these those it its they them he she we you i me my our your
""".split())

def summarize_text(text, max_sentences=5):
    # naive extractive summarization by word frequency
    sentences = [s.strip() for s in re.split(r'(?<=[\.\!\?।])\s+', text) if s.strip()]
    if not sentences:
        return ""
    # build frequencies
    freqs = {}
    for s in sentences:
        for w in re.findall(r'\w+', s.lower()):
            if w in STOPWORDS:
                continue
            freqs[w] = freqs.get(w, 0) + 1
    if not freqs:
        return " ".join(sentences[:max_sentences])

    # score sentences
    scores = []
    for s in sentences:
        score = 0
        for w in re.findall(r'\w+', s.lower()):
            if w in freqs:
                score += freqs[w]
        scores.append((score, s))

    scores.sort(reverse=True, key=lambda x: x[0])
    top = [s for _, s in scores[:max_sentences]]
    # keep the original order for readability
    top_set = set(top)
    ordered = [s for s in sentences if s in top_set]
    return " ".join(ordered)

# ---------- Translate+Summarize pipeline ----------
def process_video(video_path, target_lang):
    # 1) audio extract
    audio_path = extract_audio(video_path)

    # 2) transcribe speech
    transcript = transcribe_audio(audio_path)

    # 3) OCR on-screen text (Tesseract language: adjust if you expect Indic on screen)
    # If your on-screen text is English most of the time, keep 'eng'.
    # If it's in another language, pass its tesseract code here.
    onscreen_text = ocr_video_text(video_path, interval_sec=1.5, tesseract_lang="eng")

    # 4) combine for summary (summarize in English for quality, then translate summary)
    combined_text = (transcript + " " + onscreen_text).strip()
    base_summary = summarize_text(combined_text, max_sentences=5)

    # 5) translate transcript + summary to requested language
    translated_transcript = translate_text(transcript, target_lang)
    translated_summary = translate_text(base_summary, target_lang)

    # 6) TTS for translated transcript
    translated_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"translated_audio_{target_lang}.mp3")
    tts_from_text(translated_transcript, target_lang, translated_audio_path)

    # 7) Optional: produce dubbed video
    dubbed_video_path = os.path.splitext(video_path)[0] + f"_translated_{target_lang}.mp4"
    dubbed_video_path = merge_audio_video(video_path, translated_audio_path, output_path=dubbed_video_path)

    return {
        "audio_path": audio_path,
        "transcript": transcript,
        "onscreen_text": onscreen_text,
        "translated_transcript": translated_transcript,
        "translated_audio_path": translated_audio_path,
        "translated_summary": translated_summary,
        "dubbed_video_path": dubbed_video_path
    }

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('translate_page.html')

@app.route('/translate_page', methods=['GET', 'POST'])
def translate_page():
    if request.method == 'POST':
        upload_type = request.form.get("upload_type", "").strip().lower()
        lang = (request.form.get('language') or 'hi').strip().lower()

        # --------- Document Upload ---------
        if upload_type == "document":
            doc_file = request.files.get('document_file')
            if not doc_file or doc_file.filename == '':
                flash('No document selected', 'danger')
                return redirect(request.url)

            if not allowed_file(doc_file.filename):
                flash('Invalid document file type. Only PDF allowed.', 'danger')
                return redirect(request.url)

            filename = secure_filename(doc_file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            doc_file.save(path)

            output_path = translate_pdf(path, lang)
            return send_file(output_path, as_attachment=False)

        # --------- Video Upload ---------
        elif upload_type == "video":
            video_file = request.files.get('video_file')
            if not video_file or video_file.filename == '':
                flash('No video selected', 'danger')
                return redirect(request.url)

            filename = secure_filename(video_file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(path)

            try:
                result = process_video(path, lang)
            except Exception as e:
                print(f"Video processing error: {e}")
                flash("Failed to process/translate the video. Check server logs.", "danger")
                return redirect(request.url)

            # Render a tiny results page with links & text
            html = """
            <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 1rem 1.25rem; line-height: 1.45;">
              <h2>Video Translation Results</h2>
              <p><b>Translated audio ({{lang}}):</b> <a href="{{url_for('serve_file', filepath=result['translated_audio_path'])}}" download>Download MP3</a></p>
              <p><b>Dubbed video ({{lang}}):</b> <a href="{{url_for('serve_file', filepath=result['dubbed_video_path'])}}" download>Download MP4</a></p>
              <hr/>
              <h3>Original Transcript (detected)</h3>
              <pre style="white-space:pre-wrap;">{{result['transcript'] or '—'}}</pre>
              <h3>On-screen Text (OCR)</h3>
              <pre style="white-space:pre-wrap;">{{result['onscreen_text'] or '—'}}</pre>
              <h3>Translated Transcript ({{lang}})</h3>
              <pre style="white-space:pre-wrap;">{{result['translated_transcript'] or '—'}}</pre>
              <h3>Summary ({{lang}})</h3>
              <pre style="white-space:pre-wrap;">{{result['translated_summary'] or '—'}}</pre>
              <p><a href="{{url_for('translate_page')}}">← Go back</a></p>
            </div>
            """
            return render_template_string(html, result=result, lang=lang)

        else:
            flash("Please select upload type (Document or Video).", "danger")
            return redirect(request.url)

    return render_template('translate_page.html')

# Simple file server for generated outputs
@app.route('/serve')
def serve_file():
    filepath = request.args.get("filepath")
    if not filepath or not os.path.exists(filepath):
        flash("File not found.", "danger")
        return redirect(url_for('translate_page'))
    # guess mimetype from extension
    if filepath.lower().endswith(".mp3"):
        return send_file(filepath, as_attachment=True, mimetype="audio/mpeg")
    if filepath.lower().endswith(".mp4"):
        return send_file(filepath, as_attachment=True, mimetype="video/mp4")
    return send_file(filepath, as_attachment=True)

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
                cursor.execute(
                    "INSERT INTO users (first_name, last_name, email, username, password) VALUES (%s, %s, %s, %s, %s)",
                    tuple(data.values())
                )
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
