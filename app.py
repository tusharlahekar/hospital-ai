import os
import re
import json
import time
import math
import requests
from collections import Counter
from flask import Flask, request, jsonify, render_template

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load knowledge base
# ---------------------------------------------------------------------------
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    KB = json.load(f)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "at", "for", "with", "about",
    "what", "when", "should", "i", "we", "our", "my", "should", "do",
    "does", "did", "how", "give", "us", "tell", "me", "s", "it", "this",
    "that", "as", "by", "from", "than", "then",
}


def tokenize(text: str):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def build_doc_vector(text: str):
    return Counter(tokenize(text))


# Pre-compute a bag-of-words vector for every KB document, plus its tags
CORPUS_TEXTS = [f"{d['title']}. {d['content']} " + " ".join(d.get("tags", [])) for d in KB]
KB_VECTORS = [build_doc_vector(t) for t in CORPUS_TEXTS]


def cosine_sim(vec_a: Counter, vec_b: Counter) -> float:
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, role: str, top_k: int = 4):
    """Return top_k relevant KB docs, respecting role-based confidentiality,
    with hard-coded safety overrides for named patients / critical protocols."""

    is_privileged = role.lower() in ("admin", "hod")

    # candidate pool: strip confidential docs for non-privileged roles
    pool_idx = [i for i, d in enumerate(KB) if is_privileged or not d.get("confidential")]

    q_vec = build_doc_vector(query)
    ranked = sorted(
        ((i, cosine_sim(q_vec, KB_VECTORS[i])) for i in pool_idx),
        key=lambda x: x[1],
        reverse=True,
    )
    top_docs = [KB[i] for i, score in ranked[:top_k] if score > 0.05]

    # --- Hard safety overrides (must never depend on fuzzy similarity) ---
    q_lower = query.lower()
    safety_flags = []

    if "rajan" in q_lower:
        alert = next(d for d in KB if d.get("patient_specific") == "rajan")
        if alert not in top_docs:
            top_docs.insert(0, alert)
        else:
            top_docs.remove(alert)
            top_docs.insert(0, alert)
        safety_flags.append("PATIENT_DRUG_ALERT")

    if "padma" in q_lower:
        rec = next(d for d in KB if d.get("patient_specific") == "padma")
        if rec not in top_docs:
            top_docs.insert(0, rec)

    if "sepsis" in q_lower:
        sep = next(d for d in KB if d["title"] == "Sepsis Protocol v3")
        if sep not in top_docs:
            top_docs.insert(0, sep)
        safety_flags.append("LATEST_PROTOCOL_ENFORCED")

    if "warfarin" in q_lower or ("nsaid" in q_lower and "rajan" not in q_lower):
        warf = next(d for d in KB if d["title"] == "Warfarin-NSAID Interaction")
        if warf not in top_docs:
            top_docs.append(warf)

    return top_docs[:5], safety_flags


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Supra Multi-Specialty Hospital (Hyderabad) clinical assistant.
Rules you must follow strictly:
1. Answer ONLY using the CONTEXT provided below. Do not use outside/general medical knowledge to override it.
2. If the context contains a CRITICAL or ABSOLUTE alert relevant to the question, you MUST lead your answer with that warning, in bold, before anything else.
3. Always cite which Supra protocol/decision your answer is based on (by title).
4. If the user's role is not Admin/HOD and the context contains no permitted (non-confidential) information relevant to the question, say the information is restricted rather than guessing.
5. Keep answers concise, clinical, and actionable — this is for a doctor at the point of care, not a general audience.
6. Never invent a protocol, dose, or fact that is not present in the context.
"""


def build_context_block(docs):
    if not docs:
        return "No matching Supra Hospital protocol found in the knowledge base."
    return "\n\n".join(f"[{d['title']}]\n{d['content']}" for d in docs)


def call_groq(query, context_block, role):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User role: {role}\n\nCONTEXT:\n{context_block}\n\nQUESTION: {query}",
            },
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def fallback_answer(query, docs, safety_flags, role):
    """Deterministic, template-based answer used when no API key is configured
    or the API call fails — keeps the demo working with zero external dependency."""
    lines = []

    if "PATIENT_DRUG_ALERT" in safety_flags:
        alert = next(d for d in docs if d.get("patient_specific") == "rajan")
        lines.append(f"**\u26a0\ufe0f CRITICAL ALERT:** {alert['content']}")
        lines.append("")

    if not docs:
        lines.append(
            "I couldn't find a matching Supra Hospital protocol for this question "
            "in the knowledge base. Please check with the department HOD, or this "
            "may fall outside documented hospital policy."
        )
        return "\n".join(lines)

    lines.append("Based on Supra Hospital's own protocols:")
    for d in docs:
        if d.get("confidential") and role.lower() not in ("admin", "hod"):
            continue
        lines.append(f"\n**{d['title']}**\n{d['content']}")

    if "LATEST_PROTOCOL_ENFORCED" in safety_flags:
        lines.append(
            "\n(Note: this is the current v3 protocol — an older v2 timing "
            "would be outdated and should not be followed.)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    role = (data.get("role") or "Doctor").strip()

    if not query:
        return jsonify({"error": "empty query"}), 400

    start = time.time()
    docs, safety_flags = retrieve(query, role)
    context_block = build_context_block(
        [d for d in docs if not (d.get("confidential") and role.lower() not in ("admin", "hod"))]
    )

    used_llm = False
    answer = None

    if GROQ_API_KEY:
        try:
            answer = call_groq(query, context_block, role)
            used_llm = True
        except Exception as e:
            answer = None  # fall through to fallback

    if answer is None:
        answer = fallback_answer(query, docs, safety_flags, role)

    elapsed = round(time.time() - start, 2)

    return jsonify(
        {
            "answer": answer,
            "sources": [d["title"] for d in docs],
            "safety_flags": safety_flags,
            "used_llm": used_llm,
            "latency_sec": elapsed,
            "role": role,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
