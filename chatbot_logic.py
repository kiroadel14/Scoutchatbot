import google.generativeai as genai
import PyPDF2
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# ================== Embedding Model ==================
embed_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ================== Gemini API ==================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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

def semantic_search(question, chunks, embeddings, top_k=10):
    if embeddings.size == 0:
        return []

    q_vec = embed_model.encode(
        [question], normalize_embeddings=True
    )[0]
    scores = np.dot(embeddings, q_vec)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0.12]

# ================== Keyword Search ==================
def normalize_text(text):
    stop_words = [
        "ما", "هو", "هي", "من", "عن", "هل",
        "اذكر", "عرف", "تعريف", "عدد", "كم",
        "متى", "أين", "لماذا", "كيف",
        "ماهو", "ماهي", "مم", "بم", "بما",
        "وضح", "اشرح", "قارن", "رتب"
    ]

    words = text.replace("؟", "").split()
    return [w for w in words if w not in stop_words]

synonyms = {
    "أنواع": ["تنقسم", "تصنف", "أقسام"],
    "تعريف": ["ما هو", "ماهي", "عرّف"],
    "أهداف": ["الهدف", "غايات", "أهمية"],
    "مبادئ": ["أسس", "قيم"],
    "طريقة": ["الطريقة الكشفية", "نظام"],

    "مؤسس": ["أنشأ", "تأسست", "مؤسس الحركة"],
    "حركة": ["الحركة الكشفية", "الكشافة"],
    "غير سياسية": ["لا تنحاز", "غير حزبية"],

    "وعد": ["الوعد الكشفي", "يتعهد", "التعهد"],
    "قانون": ["قانون الكشافة", "قوانين"],

    "مراحل": ["المراحل الكشفية", "مستويات"],
    "براعم": ["البراعم"],
    "أشبال": ["الأشبال", "شبال"],
    "كشافة": ["الكشاف", "الكشافة"],
    "جوالة": ["الجوالة"],

    "طليعة": ["الطليعة", "رهط", "سداسي"],
    "عريف": ["عريف الطليعة"],
    "مسئوليات": ["مهام", "واجبات"],

    "تحية": ["التحية الكشفية"],
    "علامة": ["العلامة الكشفية"],

    "شعار": ["شعار الكشافة", "رمز"],
    "جمع": ["تقاليد الجمع", "التجمع"],
    "نداء": ["النداء", "أساليب النداء"],

    "إسعافات": ["الإسعافات الأولية"],
    "مخيمات": ["التخييم", "المخيمات"],
    "اتجاهات": ["الجهات", "تحديد الاتجاه"],
    "شفرات": ["الشفرة", "الشفرات"]
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

def is_list_question(question):
    keywords = [
        "اذكر", "عدد", "ما هي", "ما هو",
        "أنواع", "أقسام", "قانون", "مراحل"
    ]
    return any(k in question for k in keywords)

def extract_list_items(chunks):
    items = []
    for chunk in chunks:
        for line in chunk.split("\n"):
            line = line.strip()
            if (
                line.startswith("•")
                or line.startswith("-")
                or line.startswith("–")
                or line[:2].isdigit()
            ):
                items.append(line)

    seen = set()
    final_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            final_items.append(item)

    return final_items

# ================== MAIN ANSWER FUNCTION ==================
def answer_question(user_input: str) -> str:
    if not user_input.strip():
        return "❌ السؤال فارغ"

    if not pdf_chunks:
        return "❌ لم يتم تحميل المنهج"

    relevant_text = semantic_search(
        user_input, pdf_chunks, chunk_embeddings, top_k=12
    )

    if not relevant_text:
        relevant_text = find_relevant_chunks(
            user_input, pdf_chunks
        )

    if not relevant_text:
        return "⚠️ هذه المعلومة غير موجودة في المنهج"

    prompt = f"""
أنت قائد عام خبير في الكشافة.
مهمتك شرح المنهج للمخدوم.

📘 نص من المنهج:
{' '.join(relevant_text)}

❓ سؤال الطالب:
{user_input}

📌 قواعد صارمة:
- أجب من النص فقط
- لا تضف أي معلومة خارج المنهج
- اكتب الإجابة أولًا كما وردت في المنهج
- إذا كانت الإجابة تعدادًا فاذكر جميع البنود كاملة دون اختصار
- بعدها قدّم شرحًا مبسطًا للمخدوم
- لو لم تجد إجابة واضحة قل:
⚠️ هذه المعلومة غير موجودة في المنهج
- العربية فقط
"""

    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"❌ خطأ في الاتصال بالنموذج: {e}"
