# Supra Hospital AI Assistant — Design Document

**Author:** [Your name]
**Date:** [Date]
**Build time:** 90 minutes

---

## 1. Executive Summary

Supra Multi-Specialty Hospital (Hyderabad, 200+ beds, 45 doctors, 7 departments)
has accumulated years of institutional knowledge — drug protocols, patient-specific
safety alerts, departmental decisions, and lessons learned from past incidents —
that lives in people's heads, WhatsApp groups, and scattered documents. None of
it is searchable, none of it is enforced consistently, and a generic AI tool like
ChatGPT has no access to any of it and, worse, will confidently give answers that
actively contradict it.

This prototype is a **retrieval-grounded clinical assistant**: every answer is
generated (or template-assembled) strictly from Supra's own 15-entry knowledge
base, with three additional layers a doctor can actually trust — safety-critical
overrides, protocol-version enforcement, and role-based confidentiality.

---

## 2. The Core Problem: Why a Hospital Can't Just Use ChatGPT

| Failure mode | Example from this task | Consequence |
|---|---|---|
| No institutional memory | ChatGPT has never heard of "patient Rajan" | Cannot warn about his cardiac-stent NSAID contraindication — a request that has already been refused 8 times could succeed on attempt 9 |
| Generic, not Supra-specific, protocols | ChatGPT gives textbook sepsis bundle guidance | Won't know Supra tightened its lactate window from 3 hours to 1 hour in v3 — a doctor following "generic" timing is following an outdated internal standard |
| No safety-critical override | ChatGPT presents NSAIDs as a normal post-TKR option | Supra explicitly bans NSAIDs post-TKR due to bleeding risk (Dr. Vikram's decision) — ChatGPT's generic answer is actively dangerous here |
| No access control | ChatGPT will answer anyone the same way | Supra's FY2026 budget and hospital expansion plan are confidential to HOD/Admin — ChatGPT has no concept of "who is asking" |
| No traceability | ChatGPT answers with no citation to hospital policy | A doctor can't verify "is this actually our protocol?" — no audit trail for a clinical decision |
| Hallucination risk | ChatGPT will invent plausible-sounding doses/timing if unsure | In a hospital, a plausible-but-wrong answer is worse than no answer |

**The core insight:** the value of this system isn't "an LLM that knows medicine" —
ChatGPT already knows more general medicine than any of us. The value is a system
that knows **this specific hospital's rules, this specific patient's history, and
who is allowed to know what** — and that will surface the safety-critical fact
even when the user didn't explicitly ask for it.

---

## 3. Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   Browser    │────▶│  Flask /chat API  │────▶│  Retrieval Layer    │
│  (chat UI)   │◀────│                  │◀────│  (TF-IDF cosine sim)│
└─────────────┘     └──────────────────┘     └──────────┬─────────┘
                             │                            │
                             │                   ┌────────▼─────────┐
                             │                   │  15-entry JSON    │
                             │                   │  knowledge base    │
                             │                   └────────┬─────────┘
                             │                            │
                     ┌───────▼────────┐          ┌────────▼─────────┐
                     │ Safety Override │          │ Confidentiality   │
                     │ Layer (hard-    │          │ Filter (role-     │
                     │ coded rules)    │          │ based)            │
                     └───────┬────────┘          └────────┬─────────┘
                             │                            │
                             └──────────┬─────────────────┘
                                        ▼
                              ┌───────────────────┐
                              │  Generation Layer   │
                              │  (Groq LLM, or       │
                              │  offline template     │
                              │  fallback)            │
                              └───────────────────┘
```

### 3.1 Retrieval Layer
TF-IDF vectorization over the 15 knowledge-base entries, cosine similarity against
the incoming query, top-k (4) results above a relevance threshold. This is
deliberately simple and dependency-light (scikit-learn, no external vector DB)
because 15 documents doesn't need a production vector database — but the
interface is designed so it can be swapped for a proper embedding-based retriever
(e.g. sentence-transformers + FAISS/pgvector) without changing anything downstream.

### 3.2 Safety Override Layer
This is the most important architectural decision in the system: **safety-critical
facts are not left to probabilistic retrieval.** If a query mentions "Rajan," the
system deterministically force-injects his drug alert record regardless of TF-IDF
score, and flags the response as a critical alert in the UI (red banner). Similarly,
any sepsis-related query force-injects the *current* v3 protocol. Similarity search
is good at "what's related" but bad at guaranteeing "what must never be missed" —
those need explicit rules, so the two are architecturally separate layers.

### 3.3 Confidentiality Layer
Before retrieval even runs, documents flagged `confidential: true` (the FY2026
budget, the expansion plan) are stripped from the candidate pool unless the
requester's role is HOD or Admin. This happens at the retrieval stage, not just
in the prompt — so a confidential fact can never leak into context sent to the
LLM for a non-privileged user in the first place.

### 3.4 Generation Layer
Two interchangeable modes:
- **LLM mode** (Groq free API, Llama 3.3 70B): the retrieved context + a system
  prompt instructing the model to answer only from context, lead with critical
  alerts, and cite protocol titles, is sent to the model for a natural-language
  answer.
- **Offline fallback mode**: if no API key is configured, or the API call fails,
  the system deterministically assembles an answer from the retrieved documents
  using templates. This guarantees the prototype works with **zero paid tools
  and zero internet dependency**, and also means a live demo never breaks due to
  an API outage mid-presentation.

---

## 4. Data Model

Each knowledge base entry carries structured metadata beyond the raw text:

```json
{
  "title": "Patient Rajan Drug Alert",
  "content": "...",
  "department": "Cardiology",
  "confidential": false,
  "patient_specific": "rajan",
  "tags": ["rajan", "nsaid", "stent", ...]
}
```

This metadata is what makes the safety/confidentiality layers possible — a plain
text dump into a vector store would lose the ability to say "always surface this
regardless of similarity score" or "hide this from non-admins." In a real
deployment this metadata would be maintained by clinical governance / medical
records, not by whoever uploads a PDF.

---

## 5. Test Query Walkthrough

1. Post-TKR pain medication — [screenshot/output]
2. Patient Rajan prescription request — [screenshot/output]
3. DVT prophylaxis timing — [screenshot/output]
4. Sepsis protocol — [screenshot/output]
5. Mrs. Padma medication management — [screenshot/output]

---

## 6. Problems Discovered While Building

- **Similarity search alone is not enough for safety.** Early in development,
  a generic post-TKR pain query only surfaced the Rajan alert because it happened
  to share vocabulary ("NSAID," "surgical") — it would have been silent for a
  differently-worded query. This is why the hard-coded safety override layer
  exists as a separate, non-probabilistic mechanism (Section 3.2). **Lesson:**
  in a clinical system, "usually retrieves the right thing" is not good enough
  for anything safety-critical — those need explicit, testable rules.
- **Confidentiality has to be enforced before generation, not just prompted.**
  Initially the confidentiality rule was just an instruction in the system
  prompt ("don't reveal confidential info to non-admins"). This is fragile — a
  differently-phrased question or a capable-enough model could still leak it if
  the confidential text is sitting in the context window. Moving the filter to
  the retrieval stage (so confidential documents are never even fetched for a
  non-privileged role) closes that gap architecturally instead of relying on
  the model's obedience.
- **Version drift is a silent danger.** The knowledge base itself contains an
  example of this: the sepsis protocol went from a 3-hour to a 1-hour lactate
  window. A naive retrieval system has no concept of "supersedes" — it would
  happily surface an old cached protocol. This system currently handles it by
  only keeping the current version in the KB, but a real system needs explicit
  versioning (Section 7).
- **90 minutes is not enough time for a production vector DB, auth system, or
  EHR integration** — this prototype makes deliberate scope cuts (see Section 8)
  to prove the *concept* (grounded, safety-aware, access-controlled answers) in
  the time available, rather than partially building infrastructure that
  wouldn't be demoable.

---

## 7. What I'd Add With More Time

**Short term (next few days):**
- Proper embedding-based retrieval (sentence-transformers) instead of TF-IDF,
  for better recall on paraphrased questions.
- Explicit document versioning/supersession (e.g. "Sepsis Protocol v2" marked
  `superseded_by: v3` so old versions are never retrievable, not just absent).
- Real authentication (not a role dropdown) tied to hospital staff directory.
- Structured patient records (not just free-text KB entries) so patient alerts
  are pulled from an actual patient ID lookup, not string-matching "rajan" in
  the query text.
- Logging every query + answer + sources for clinical audit/liability purposes.

**Medium term:**
- Integration with the hospital's actual EHR for live patient context (current
  medications, allergies, recent labs) rather than a static knowledge base.
- A feedback loop where doctors can flag a wrong/outdated answer, routed to
  the relevant HOD for review — this is how the knowledge base stays current.
- Department-specific fine-tuning of the retrieval ranking (an ICU query and an
  Ortho query should weight department-relevant documents differently).
- Multi-turn conversation memory (a doctor should be able to ask a follow-up
  like "what if VAS is 8?" without repeating full context).

**Long term:**
- A formal "critical alert" authoring workflow so any clinician (not just a
  developer) can register a hard safety override like the Rajan alert.
- Mobile-first interface for ward rounds.
- Governance dashboard showing which protocols are most-queried and where staff
  disagree with the system's answer — signal for where documentation is unclear.

---

## 8. Explicit Scope Cuts (and why)

| Cut | Reason |
|---|---|
| No real authentication | Role dropdown simulates it; real auth is a separate, well-understood problem not worth the build time in 90 minutes |
| No vector database | 15 documents don't need one; TF-IDF proves the retrieval concept at this scale |
| No persistent chat history / database | Not needed to demonstrate the core grounding + safety concept |
| No multi-patient record system | Patient alerts are matched by name in the KB, not a real EHR lookup — a real deployment would need this, but building an EHR mock wasn't the highest-value use of the time |

---

## 9. Why This Is "Measurably Better" Than ChatGPT

Concretely, for the same 5 test queries, this system does what ChatGPT structurally
cannot:
1. Cites a specific Supra decision-maker and date (Dr. Vikram, Jan 2025) — ChatGPT
   cannot cite an internal decision it has never seen.
2. Surfaces a named patient's contraindication unprompted — ChatGPT has zero
   patient records.
3. Enforces the *current* internal protocol version — ChatGPT has no way to know
   Supra's timing was ever tightened.
4. Refuses to answer a confidential financial question for a non-privileged role —
   ChatGPT applies the same answer to everyone.
5. Every answer traces back to a named, auditable source document — ChatGPT's
   answers are not traceable to anything.

This is the difference between "an assistant that knows medicine" and "an
assistant that knows *this hospital*" — and only the second one is safe and
useful at the point of care.

---

## 10. Conclusion

The prototype demonstrates that the meaningful difference between a hospital AI
assistant and raw ChatGPT isn't a bigger or smarter model — it's **grounding,
safety-critical determinism, and access control** layered around retrieval. All
three are architecturally simple to add, cost nothing (this entire system runs
on free tools), and are the difference between a system a doctor would use once
out of curiosity and one they'd trust at the point of care.
