import google.generativeai as genai
import PyPDF2
import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

# ================== Flask ==================
app = Flask(__name__)

# ================== Embedding Model ==================
embed_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ================== Gemini API ==================
genai.configure(api_key=os.getenv("AIzaSyDsYHpZEbjkq1Fp-gbPTkaSi3bb7Hx5Kh4"))
model = genai.GenerativeModel("gemini-3-flash-preview")
chat = model.start_chat(history=[])

# ================== PDF PATH ==================
PDF_PATH = os.path.join(os.path.dirname(__file__), "scout.pdf")

# ================== Load PDF ==================
def load_pdf_chunks(path, chunk_size=800):
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

    chunks = []
    current = ""
    for line in text.split("\n"):
        current += line + " "
        if len(current) >= chunk_size:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())

    return chunks

try:
    pdf_chunks = load_pdf_chunks(PDF_PATH)
    print("PDF Loaded:", len(pdf_chunks))
except Exception as e:
    print("PDF ERROR:", e)
    pdf_chunks = []

# ================== Embeddings ==================
if pdf_chunks:
    chunk_embeddings = embed_model.encode(
        pdf_chunks, normalize_embeddings=True
    )
else:
    chunk_embeddings = np.array([])

def semantic_search(question, chunks, embeddings, top_k=5):
    if embeddings.size == 0:
        return []

    q_vec = embed_model.encode(
        [question], normalize_embeddings=True
    )[0]
    scores = np.dot(embeddings, q_vec)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0.2]

# ================== Keyword Search ==================
def normalize_text(text):
    stop_words = [
        "ما", "هو", "هي", "من", "عن",
        "أنواع", "اذكر", "عرف", "تعريف"
    ]
    words = text.replace("؟", "").split()
    return [w for w in words if w not in stop_words]

synonyms = {
    "أنواع": ["تنقسم", "تصنف"],
    "مؤسس": ["أنشأ", "مؤسس الحركة"],
    "واجبات": ["يلتزم", "مسؤوليات"],
    "وعد": ["يتعهد", "التعهد"],
}

def expand_keywords(words):
    expanded = set(words)
    for w in words:
        if w in synonyms:
            expanded.update(synonyms[w])
    return expanded

def find_relevant_chunks(question, chunks, max_chunks=5):
    keywords = expand_keywords(normalize_text(question))
    scored_chunks = []

    for chunk in chunks:
        score = 0
        for word in keywords:
            if word in chunk:
                score += chunk.count(word) * 2
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:max_chunks]]

# ================== API Endpoint ==================
@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "⚠️ من فضلك اكتب سؤالًا"})

    if not pdf_chunks:
        return jsonify({"reply": "❌ لم يتم تحميل المنهج."})

    relevant_text = semantic_search(
        user_input, pdf_chunks, chunk_embeddings
    )

    if not relevant_text:
        relevant_text = find_relevant_chunks(
            user_input, pdf_chunks
        )

    if not relevant_text:
        return jsonify({
            "reply": "⚠️ هذه المعلومة غير موجودة في المنهج."
        })

    prompt = f"""
أنت مدرس كشافة خبير.
مهمتك شرح المنهج للطلاب.

📘 نص من المنهج:
{' '.join(relevant_text)}

❓ سؤال الطالب:
{user_input}

📌 قواعد صارمة:
- أجب من النص فقط
- لا تضف أي معلومة خارج المنهج
- اكتب الإجابة أولًا كما وردت في المنهج
- بعدها قدّم شرحًا مبسطًا للطالب
- لو لم تجد إجابة واضحة قل:
⚠️ هذه المعلومة غير موجودة في المنهج
- العربية فقط
"""

    try:
        response = chat.send_message(prompt)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"❌ خطأ: {e}"})


if __name__ == "__main__":
    app.run(debug=True)
