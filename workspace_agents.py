"""
workspace_agents.py
-------------------
نظام الوكلاء المتعددين الخاص بصفحة المسودة / مساحة العمل.

يعيد هذا الملف استخدام ملفات وكيل خدمة العملاء الموجودة كما هي، دون تعديلها:

    app/rag.py    -> استرجاع مقالات قاعدة المعرفة (الأقسام + تشابه جيب التمام)
    app/tools.py  -> أدوات فعلية (search_knowledge_tool, extract_ticket_reference)
    app/agent.py  -> detect_language + rule_based_decision (محرّك القرار الرباعي)
    app/prompts.py-> مطالبات الوكلاء الثلاثة
    app/models.py -> AgentDecision

الوكلاء الثلاثة:

  1) ResearcherAgent      (rag.py + tools.py)
     يجلب أعلى k مقالات من مخزن المستندات، ويحدّد أي جُمل المسودة مدعومة
     بمصدر فعلي، وأيها يذكر سياسة غير موجودة في قاعدة المعرفة (خطر!).

  2) GrammarToneAgent     (agent.py + بنك الكلمات)
     يصحّح الإملاء مقابل words_alpha.txt، ويصلح النحو، ويحوّل النبرة الجافة
     إلى نبرة خدمة عملاء مهنية ومهذّبة.

  3) LeadSynthesizerAgent
     يدمج المخرجات في مسودة نهائية جاهزة للإرسال + سجل تغييرات +
     خيارات ردود مقترحة (مبنية على قرار الوكيل الرباعي).

التنسيق (Orchestration) عبر LangGraph StateGraph بنفس أسلوب app/agent.py:

    research -> grammar_tone -> synthesize -> END

كل وكيل يعمل بوضعين: OpenAI عند تفعيله، ورجوع آمن (fallback) لوضع القواعد
عند أي فشل، فلا تتعطّل نقطة النهاية أبدًا.
"""

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, TypedDict

from langgraph.graph import StateGraph, END

from config import settings
from models import AgentDecision
from app import rag as rag_module
from app import tools as tools_module
from agent import detect_language, rule_based_decision
from prompts import (
    RESEARCHER_PROMPT,
    GRAMMAR_TONE_PROMPT,
    SYNTHESIZER_PROMPT,
    build_researcher_prompt,
    build_grammar_tone_prompt,
    build_synthesizer_prompt,
)

# ---------------------------------------------------------------------------
# إعدادات
# ---------------------------------------------------------------------------
TOP_K_ARTICLES = 4          # عدد مقالات قاعدة المعرفة المسترجعة
CITE_THRESHOLD = 0.12       # حد تشابه الجملة مع المقال لاعتبارها مدعومة
_LETTERS = "abcdefghijklmnopqrstuvwxyz"

_WORD_BANK_PATHS = [
    getattr(settings, "word_bank_path", None),
    "data/words_alpha.txt", "words_alpha.txt", "app/data/words_alpha.txt",
]

# تصحيحات إملائية صريحة لها أولوية على القاموس (تحسم الحالات الملتبسة)
COMMON_FIXES = {
    "colected": "collected", "helo": "hello", "teh": "the", "recieve": "receive",
    "recieved": "received", "seperate": "separate", "definately": "definitely",
    "wich": "which", "becuase": "because", "thier": "their", "adress": "address",
    "untill": "until", "begining": "beginning", "beleive": "believe",
    "sucessful": "successful", "aplogies": "apologies", "inconvienience": "inconvenience",
    "cant": "can't", "dont": "don't", "doesnt": "doesn't", "wont": "won't",
    "isnt": "isn't", "im": "I'm", "youre": "you're",
    # أخطاء شائعة لها أكثر من مرشّح في القاموس، فنحسمها صريحةً
    "acount": "account", "acounts": "accounts", "adres": "address",
    "recive": "receive", "pasword": "password", "custmer": "customer",
    "custommer": "customer", "verifcation": "verification", "refund ": "refund ",
    "servcie": "service", "suport": "support", "tommorrow": "tomorrow",
}

# ---------------------------------------------------------------------------
# قواعد النبرة
#   TONE_SUBS  : استبدالات محلية آمنة (لا تُخلّ بالنحو) -> (نمط, بديل, وصف)
#   TONE_FLAGS : عبارات تحتاج إعادة صياغة بشرية/نموذجية -> نقترحها ولا نستبدلها
#                (الاستبدال الآلي لجملة رفض يُنتج نصًا مشوّهًا)
# ---------------------------------------------------------------------------
TONE_SUBS_EN = [
    (r"\byou must\b", "could you please", "Replaced the command “you must” with a courteous request"),
    (r"\byou have to\b", "could you please", "Replaced “you have to” with a courteous request"),
    (r"\byou need to\b", "please", "Softened “you need to” to “please”"),
    (r"\byour mistake\b", "what happened", "Removed blame (“your mistake”)"),
    (r"\bcalm down\b", "I understand this is frustrating", "Replaced “calm down” with empathy"),
    (r"\bobviously\b\s*", "", "Removed condescending “obviously”"),
]
TONE_SUBS_AR = [
    (r"\bيجب عليك\b", "نرجو منك", "استبدال الأمر «يجب عليك» بصيغة مهذّبة"),
    (r"\bعليك أن\b", "نرجو منك أن", "استبدال «عليك أن» بصيغة مهذّبة"),
    (r"\bخطؤك\b", "ما حدث", "إزالة إلقاء اللوم على العميل"),
    (r"\bاهدأ\b", "أتفهّم أن الأمر مزعج", "استبدال «اهدأ» بعبارة تعاطف"),
]
TONE_FLAGS_EN = [
    (r"\bwe can'?t do (that|it)\b", "Blunt refusal — state what you CAN do and the next step instead"),
    (r"\bthat'?s not possible\b", "Blunt refusal — offer the closest available alternative"),
    (r"\byou failed to\b", "Blaming phrasing — describe the situation neutrally"),
    (r"\bno\.\s", "Bare refusal — soften and explain the reason"),
]
TONE_FLAGS_AR = [
    (r"\bلا يمكننا ذلك\b", "رفض مباشر — اذكر ما يمكن تقديمه والخطوة التالية"),
    (r"\bغير ممكن\b", "رفض مباشر — اقترح أقرب بديل متاح"),
]

# ---------------------------------------------------------------------------
# فحص الامتثال: قواعد prompts.py تمنع طلب كلمة المرور/OTP/بطاقة الدفع.
# إن ظهرت في المسودة نرفع تحذيرًا بارزًا بدل "تلطيف" الطلب الممنوع.
# ---------------------------------------------------------------------------
COMPLIANCE_PATTERNS = [
    (r"\b(password|passcode)\b|كلمة\s*(ال)?مرور|كلمة\s*السر",
     "asks the customer for their password",
     "تطلب المسودة كلمة مرور العميل"),
    (r"\b(otp|verification code|one[- ]time code)\b|رمز\s*التحقق",
     "asks the customer for a verification code (OTP)",
     "تطلب المسودة رمز التحقق (OTP)"),
    (r"\b(card number|cvv|credit card|bank card)\b|رقم\s*البطاقة|بطاقة\s*بنكية",
     "asks the customer for payment card details",
     "تطلب المسودة بيانات بطاقة الدفع"),
]
# طلب فعلي (لا مجرد ذكر) — نحتاج فعل طلب قريبًا من الكلمة الحساسة
_ASK_VERBS = re.compile(
    r"\b(send|share|give|provide|tell|confirm|reply with|enter)\b|"
    r"أرسل|شارك|أعطِ|زوّدنا|أخبرنا|أكّد|اكتب", re.IGNORECASE)

_POLITE_OPENER = {
    "en": "Thank you for reaching out, and I'm sorry for the trouble.",
    "ar": "شكرًا لتواصلك معنا، ونعتذر عن الإزعاج.",
}
_NEXT_STEP = {
    "answer":        {"en": "I hope this resolves it — please let me know if anything is still unclear.",
                      "ar": "أرجو أن يكون هذا قد حلّ الأمر، ولا تتردد في إخبارنا إن بقي أي استفسار."},
    "create_ticket": {"en": "I've opened a support ticket for you and our team will follow up shortly.",
                      "ar": "فتحنا لك تذكرة دعم وسيتابع فريقنا معك قريبًا."},
    "escalate":      {"en": "I'm raising your case to a specialist right away, and they will contact you as a priority.",
                      "ar": "نرفع حالتك إلى مختص فورًا وسيتواصل معك على سبيل الأولوية."},
    "check_ticket":  {"en": "You can reply here any time for another update on your ticket.",
                      "ar": "يمكنك مراسلتنا في أي وقت للحصول على تحديث آخر عن تذكرتك."},
}


# ---------------------------------------------------------------------------
# بنك الكلمات (words_alpha.txt)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_word_bank() -> frozenset:
    """يحمّل بنك الكلمات مرة واحدة إلى frozenset. يزيل CRLF. فارغ إن لم يوجد."""
    for path in _WORD_BANK_PATHS:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return frozenset(w.strip().lower() for w in f if w.strip())
    return frozenset()


def _edits1(word: str) -> Set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in _LETTERS]
    inserts = [a + c + b for a, b in splits for c in _LETTERS]
    return set(deletes + transposes + replaces + inserts)


def _edits1_known(word: str, bank: frozenset) -> Set[str]:
    return {e for e in _edits1(word) if e in bank}


def bank_candidates(word: str, bank: frozenset, limit: int = 6) -> List[str]:
    """أقرب كلمات القاموس (مسافة 1 ثم 2) — تُمرَّر للنموذج كتلميحات."""
    low = word.lower()
    e1 = _edits1_known(low, bank)
    if e1:
        return sorted(e1)[:limit]
    e2: Set[str] = set()
    for e in _edits1(low):
        e2 |= _edits1_known(e, bank)
    return sorted(e2)[:limit]


def word_bank_hints(text: str, bank: frozenset, max_words: int = 40) -> List[tuple]:
    if not bank:
        return []
    seen, hints = set(), []
    for w in re.findall(r"[A-Za-z]+", text):
        low = w.lower()
        if len(low) <= 2 or low in seen:
            continue
        seen.add(low)
        if low in bank:
            continue
        hints.append((w, bank_candidates(low, bank)))
        if len(hints) >= max_words:
            break
    return hints


def _match_case(orig: str, fix: str) -> str:
    if len(orig) > 1 and orig.isupper():
        return fix.upper()
    if orig[:1].isupper():
        return fix[:1].upper() + fix[1:]
    return fix


# ---------------------------------------------------------------------------
# أدوات نصية مشتركة
# ---------------------------------------------------------------------------
def _tidy_text(text: str) -> str:
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\s+([,.;:!?؟،])", r"\1", t)
    t = re.sub(r"([,;:،])(?=\S)", r"\1 ", t)
    t = re.sub(r"([.!?؟])(?=[A-Za-z\u0600-\u06FF])", r"\1 ", t)
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return t.strip()


def split_sentences(text: str) -> List[str]:
    """تقسيم موحّد للجُمل حتى تتوافق فهارس الباحث مع مواضع التوثيق."""
    out: List[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        out.extend(s for s in re.split(r"(?<=[.!؟?])\s+", line) if s.strip())
    return out


def _R(language: str, ar: str, en: str) -> str:
    return ar if language == "ar" else en


def _use_llm() -> bool:
    return bool(settings.use_openai and settings.openai_api_key)


def _chat_json(system: str, user: str) -> dict:
    """نداء OpenAI يعيد JSON. يرفع استثناءً عند الفشل ليتولّى الـ fallback."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.1,  # محرّر دقيق وحتمي
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    content = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    return json.loads(match.group(0) if match else content)


# ---------------------------------------------------------------------------
# حالة الرسم البياني
# ---------------------------------------------------------------------------
class DraftState(TypedDict, total=False):
    # المدخلات
    raw_draft: str
    customer_id: Optional[str]
    language: str

    # مخرجات وكيل البحث
    articles: List[dict]          # [{n, title, content, score, source}]
    sentences: List[str]
    supported: List[dict]         # [{i, n, why}]
    unsupported: List[dict]       # [{i, risk}]

    # مخرجات وكيل اللغة والنبرة
    polished_text: str
    grammar_changes: List[dict]

    # مخرجات الوكيل القائد
    final_draft: str
    suggestions: List[dict]
    citation_changes: List[dict]
    cited: List[int]

    # عام
    decision: dict                # قرار الوكيل الرباعي (اقتراح الإجراء التالي)
    agent_reports: List[dict]
    used_llm: bool


# ===========================================================================
# 1) وكيل البحث — Researcher / RAG Agent   (rag.py + tools.py)
# ===========================================================================
def retrieve_articles(query: str, top_k: int = TOP_K_ARTICLES) -> List[dict]:
    """
    يسترجع أعلى k أقسام من مخزن المستندات باستخدام نفس منطق app/rag.py
    (الأقسام المخزّنة مؤقتًا + تشابه جيب التمام)، بدل نتيجة واحدة فقط.
    """
    sections = rag_module._get_sections_cached(settings.knowledge_base_path)
    q_tf = rag_module._term_frequency(rag_module._tokenize(query))
    if not q_tf or not sections:
        return []

    scored = []
    for sec in sections:
        score = rag_module._cosine_similarity(q_tf, sec.tf)
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    kb_name = settings.knowledge_base_path.split("/")[-1]
    return [
        {"n": i + 1, "title": sec.title, "content": sec.content,
         "score": round(score, 4), "source": f"{kb_name}#{sec.title}"}
        for i, (score, sec) in enumerate(scored[:top_k])
    ]


def _research_rule_based(sentences: List[str], articles: List[dict]) -> Tuple[List[dict], List[dict]]:
    """مطابقة الجُمل بالمقالات عبر تشابه جيب التمام (بدون نموذج لغوي)."""
    art_tf = [(a, rag_module._term_frequency(rag_module._tokenize(f"{a['title']} {a['content']}")))
              for a in articles]
    supported, unsupported = [], []
    # كلمات تدل على أن الجملة تقرّر سياسة/مدة/مبلغًا (تستوجب دعمًا بمصدر)
    policy_re = re.compile(
        r"\b(polic\w*|refund|guarantee|within|days?|hours?|fee|charge|free|"
        r"سياسة|استرجاع|ضمان|خلال|يوم|أيام|ساعة|رسوم|مجان\w*)\b", re.IGNORECASE)

    for i, sen in enumerate(sentences):
        if re.search(r"\[\d+\]", sen):
            continue
        s_tf = rag_module._term_frequency(rag_module._tokenize(sen))
        best, best_score = None, 0.0
        for a, tf in art_tf:
            sc = rag_module._cosine_similarity(s_tf, tf)
            if sc > best_score:
                best_score, best = sc, a
        if best and best_score >= CITE_THRESHOLD:
            supported.append({"i": i, "n": best["n"],
                              "why": f"cosine {round(best_score, 2)} vs “{best['title']}”"})
        elif policy_re.search(sen):
            unsupported.append({"i": i, "risk": "states a policy/timeframe with no supporting article"})
    return supported, unsupported


def researcher_node(state: DraftState) -> DraftState:
    draft = state["raw_draft"]
    language = state["language"]
    articles = retrieve_articles(draft)
    sentences = split_sentences(draft)

    supported: List[dict] = []
    unsupported: List[dict] = []
    used_llm = False

    if articles:
        if _use_llm():
            try:
                data = _chat_json(RESEARCHER_PROMPT,
                                  build_researcher_prompt(sentences, articles, language))
                valid_ns = {a["n"] for a in articles}
                for item in data.get("supported", []):
                    try:
                        i, n = int(item["i"]), int(item["n"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 0 <= i < len(sentences) and n in valid_ns:
                        supported.append({"i": i, "n": n, "why": item.get("why", "")})
                for item in data.get("unsupported", []):
                    try:
                        i = int(item["i"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 0 <= i < len(sentences):
                        unsupported.append({"i": i, "risk": item.get("risk", "")})
                used_llm = True
            except Exception:
                supported, unsupported = _research_rule_based(sentences, articles)
        else:
            supported, unsupported = _research_rule_based(sentences, articles)

    rationale = _R(language,
        f"استرجعت {len(articles)} مقالًا من مخزن المستندات، وفحصت {len(sentences)} جملة: "
        f"{len(supported)} مدعومة بمصدر و{len(unsupported)} تحتاج تحقّقًا.",
        f"Retrieved {len(articles)} knowledge-base article(s) and checked {len(sentences)} sentence(s): "
        f"{len(supported)} backed by a source, {len(unsupported)} needing verification.")

    reports = list(state.get("agent_reports", []))
    reports.append({"agent": "researcher", "rationale": rationale,
                    "changes": [], "articles": [a["title"] for a in articles]})

    return {"articles": articles, "sentences": sentences, "supported": supported,
            "unsupported": unsupported, "agent_reports": reports,
            "used_llm": state.get("used_llm", False) or used_llm}


# ===========================================================================
# 2) وكيل اللغة والنبرة — Grammar & Tone Agent   (agent.py + بنك الكلمات)
# ===========================================================================
def _grammar_tone_rule_based(text: str, language: str, bank: frozenset) -> Tuple[str, List[dict]]:
    changes: List[dict] = []
    seen: Set[tuple] = set()

    def _record(w: str, to: str) -> None:
        if (w, to) not in seen:
            seen.add((w, to))
            changes.append({"type": "typo", "from": w, "to": to})

    def _fix(m: re.Match) -> str:
        w = m.group(0)
        low = w.lower()
        if low in COMMON_FIXES:                       # 1) تصحيحات صريحة
            to = _match_case(w, COMMON_FIXES[low]); _record(w, to); return to
        if not bank or len(low) <= 2 or low in bank:  # 2) كلمة صحيحة
            return w
        cands = _edits1_known(low, bank)              # 3) مرشّح وحيد فقط
        if len(cands) == 1:
            to = _match_case(w, next(iter(cands))); _record(w, to); return to
        return w

    spell_fixed = re.sub(r"[A-Za-z]+", _fix, text)

    tidied = _tidy_text(spell_fixed)
    if tidied != spell_fixed:
        changes.append({"type": "grammar", "detail": _R(language,
            "تصحيح المسافات وعلامات الترقيم وبدايات الجُمل",
            "Corrected spacing, punctuation & sentence casing")})

    # نبرة خدمة العملاء: استبدالات آمنة فقط
    toned = tidied
    for pattern, repl, label in (TONE_SUBS_AR if language == "ar" else TONE_SUBS_EN):
        new = re.sub(pattern, repl, toned, flags=re.IGNORECASE)
        if new != toned:
            changes.append({"type": "tone", "detail": label})
            toned = new

    # عبارات تحتاج إعادة صياغة: نرفعها كتنبيه بدل تشويه النص آليًا
    for pattern, label in (TONE_FLAGS_AR if language == "ar" else TONE_FLAGS_EN):
        if re.search(pattern, toned, flags=re.IGNORECASE):
            changes.append({"type": "risk", "detail": label})

    # فحص الامتثال: طلب كلمة مرور/OTP/بطاقة ممنوع في قواعد المشروع
    for pattern, en_msg, ar_msg in COMPLIANCE_PATTERNS:
        m = re.search(pattern, toned, flags=re.IGNORECASE)
        if not m:
            continue
        window = toned[max(0, m.start() - 60): m.end() + 60]
        if _ASK_VERBS.search(window):
            changes.append({"type": "compliance", "detail": _R(language,
                f"تحذير: {ar_msg} — وهذا ممنوع في سياسة الدعم؛ يجب إزالة الطلب.",
                f"Warning: the draft {en_msg} — this is prohibited by support policy and must be removed.")})

    toned = _tidy_text(toned)

    # افتتاحية مهذّبة إن بدأت المسودة بجفاف
    opener = _POLITE_OPENER[language if language in _POLITE_OPENER else "en"]
    first = (split_sentences(toned) or [""])[0].lower()
    if toned and not any(k in first for k in
                         ("thank", "sorry", "apolog", "hello", "hi ", "شكرًا", "نعتذر", "مرحبًا", "أهلًا")):
        toned = f"{opener} {toned}"
        changes.append({"type": "tone", "detail": _R(language,
            "إضافة افتتاحية مهذّبة تُقرّ بموقف العميل",
            "Added a courteous opener acknowledging the customer")})

    return toned, changes


def grammar_tone_node(state: DraftState) -> DraftState:
    text, language = state["raw_draft"], state["language"]
    bank = load_word_bank()
    polished: Optional[str] = None
    changes: List[dict] = []
    used_llm = False

    if _use_llm():
        try:
            data = _chat_json(GRAMMAR_TONE_PROMPT,
                              build_grammar_tone_prompt(text, language, word_bank_hints(text, bank)))
            polished = data.get("polished_text") or None
            changes = data.get("changes") or []
            used_llm = polished is not None
        except Exception:
            polished = None

    if polished is None:
        polished, changes = _grammar_tone_rule_based(text, language, bank)

    n_typo = sum(1 for c in changes if c.get("type") == "typo")
    n_tone = sum(1 for c in changes if c.get("type") == "tone")
    n_gram = sum(1 for c in changes if c.get("type") == "grammar")
    rationale = _R(language,
        f"تحقّقت من كل كلمة مقابل قاموس من {len(bank):,} كلمة: {n_typo} تصحيح إملائي، "
        f"{n_gram} تصحيح نحوي، و{n_tone} تعديل نبرة نحو أسلوب خدمة عملاء مهني.",
        f"Verified every word against a {len(bank):,}-word dictionary: {n_typo} spelling fix(es), "
        f"{n_gram} grammar fix(es), {n_tone} tone adjustment(s) toward professional support style.")

    reports = list(state.get("agent_reports", []))
    reports.append({"agent": "grammar_tone", "rationale": rationale, "changes": changes})

    return {"polished_text": polished, "grammar_changes": changes, "agent_reports": reports,
            "used_llm": state.get("used_llm", False) or used_llm}


# ===========================================================================
# 3) الوكيل القائد — Lead Synthesizer Agent
# ===========================================================================
def _suggest_next_action(state: DraftState) -> AgentDecision:
    """
    يعيد استخدام محرّك القرار الرباعي في app/agent.py لاقتراح الإجراء التالي
    المناسب للمسودة (answer / create_ticket / escalate / check_ticket).
    """
    draft = state["raw_draft"]
    articles = state.get("articles", [])
    best = articles[0] if articles else None
    rag_result = rag_module.RagResult(
        found=bool(best and best["score"] >= settings.min_rag_score),
        answer=best["content"] if best else None,
        source=best["source"] if best else None,
        section_title=best["title"] if best else None,
        confidence=best["score"] if best else 0.0,
    )
    ticket_ref = tools_module.extract_ticket_reference(draft)
    return rule_based_decision(draft, state["language"], rag_result, ticket_ref)


def _place_citations(text: str, supported: List[dict], articles: List[dict],
                     language: str) -> Tuple[str, List[dict], List[int]]:
    """يضع وسوم [n] على الجُمل المدعومة ويبني قائمة المراجع."""
    by_n = {a["n"]: a for a in articles}
    changes: List[dict] = []
    cited: Set[int] = set()
    n_for_index = {s["i"]: s["n"] for s in supported}

    idx, out_lines = 0, []
    for line in text.split("\n"):
        if not line.strip():
            out_lines.append(line)
            continue
        new_sens = []
        for sen in re.split(r"(?<=[.!؟?])\s+", line):
            if not sen.strip():
                continue
            n = n_for_index.get(idx)
            idx += 1
            if n and not re.search(r"\[\d+\]", sen):
                cited.add(n)
                changes.append({"type": "citation", "n": n,
                                "detail": by_n.get(n, {}).get("title", "")})
                if re.search(r"[.!؟?][\"']?\s*$", sen):
                    sen = re.sub(r"([.!؟?][\"']?)\s*$", rf" [{n}]\1", sen)
                else:
                    sen = f"{sen} [{n}]"
            new_sens.append(sen)
        out_lines.append(" ".join(new_sens))
    body = "\n".join(out_lines)

    if cited:
        head = "المصادر" if language == "ar" else "Sources"
        lines = [f"[{n}] {by_n[n]['title']}" for n in sorted(cited) if n in by_n]
        body = f"{body}\n\n{head}\n" + "\n".join(lines)
    return body, changes, sorted(cited)


def synthesize_node(state: DraftState) -> DraftState:
    language = state["language"]
    polished = state["polished_text"]
    articles = state.get("articles", [])
    supported = state.get("supported", [])
    unsupported = state.get("unsupported", [])

    decision = _suggest_next_action(state)
    final_text: Optional[str] = None
    suggestions: List[dict] = []
    llm_rationale = ""
    used_llm = False

    if _use_llm():
        try:
            data = _chat_json(SYNTHESIZER_PROMPT, build_synthesizer_prompt(
                polished, supported, unsupported, articles, decision.action, language))
            final_text = data.get("final_draft") or None
            suggestions = [s for s in (data.get("suggestions") or []) if s.get("text")]
            llm_rationale = data.get("rationale", "")
            used_llm = final_text is not None
        except Exception:
            final_text = None

    citation_changes: List[dict] = []
    cited: List[int] = []
    if final_text is None:
        # وضع القواعد: توثيق + خطوة تالية مهذّبة متوافقة مع القرار
        final_text, citation_changes, cited = _place_citations(polished, supported, articles, language)
        step = _NEXT_STEP.get(decision.action, _NEXT_STEP["answer"])[
            language if language in ("ar", "en") else "en"]
        if step and step not in final_text:
            head, sep, tail = final_text.partition("\n\n")
            body = head if sep else final_text
            if body and not re.search(r"[.!?؟]['\"]?\s*$", body.rstrip()):
                body = body.rstrip() + "."          # لا تلصق الخطوة التالية بجملة ناقصة
            final_text = f"{body} {step}{sep}{tail}" if sep else f"{body} {step}"
    else:
        cited = [int(n) for n in set(re.findall(r"\[(\d+)\]", final_text))]
        citation_changes = [{"type": "citation", "n": n,
                             "detail": next((a["title"] for a in articles if a["n"] == n), "")}
                            for n in sorted(cited)]

    # خيارات ردود مقترحة (تُعرض بجوار المحرّر ويمكن إدراجها بنقرة)
    if not suggestions:
        for a in articles[:3]:
            suggestions.append({
                "title": a["title"],
                "text": (a["content"].strip().split("\n")[0])[:280],
                "n": a["n"],
            })
    suggestions.insert(0, {
        "title": _R(language, "الإجراء المقترح", "Recommended action"),
        "text": _NEXT_STEP.get(decision.action, _NEXT_STEP["answer"])[
            language if language in ("ar", "en") else "en"],
        "action": decision.action,
    })

    aggregated = list(state.get("grammar_changes", [])) + citation_changes
    n_typo = sum(1 for c in aggregated if c.get("type") == "typo")
    n_gram = sum(1 for c in aggregated if c.get("type") == "grammar")
    n_tone = sum(1 for c in aggregated if c.get("type") == "tone")
    n_cite = sum(1 for c in aggregated if c.get("type") == "citation")
    rationale = llm_rationale or _R(language,
        f"دمجت مخرجات الوكلاء في مسودة نهائية: {n_typo} إملاء، {n_gram} نحو، {n_tone} نبرة، "
        f"{n_cite} توثيق. الإجراء المقترح: {decision.action} — {decision.reason}",
        f"Merged the agents' output into a send-ready draft: {n_typo} spelling, {n_gram} grammar, "
        f"{n_tone} tone, {n_cite} citation. Recommended action: {decision.action} — {decision.reason}")

    reports = list(state.get("agent_reports", []))
    reports.append({"agent": "lead_synthesizer", "rationale": rationale, "changes": citation_changes})

    return {"final_draft": final_text, "suggestions": suggestions,
            "citation_changes": citation_changes, "cited": cited,
            "decision": decision.model_dump(), "agent_reports": reports,
            "used_llm": state.get("used_llm", False) or used_llm}


# ===========================================================================
# بناء الرسم البياني وتشغيله
# ===========================================================================
def build_draft_graph():
    """research -> grammar_tone -> synthesize -> END"""
    graph = StateGraph(DraftState)
    graph.add_node("research", researcher_node)
    graph.add_node("grammar_tone", grammar_tone_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "grammar_tone")
    graph.add_edge("grammar_tone", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


_DRAFT_GRAPH = None


def _graph():
    global _DRAFT_GRAPH
    if _DRAFT_GRAPH is None:
        _DRAFT_GRAPH = build_draft_graph()
    return _DRAFT_GRAPH


def enhance_draft(draft: str, language: Optional[str] = None,
                  customer_id: Optional[str] = None) -> dict:
    """
    نقطة الدخول التي يستخدمها main.py / الراوتر.

    تعيد Dictionary جاهزًا لملء نموذج الاستجابة، ويحتوي مفتاح
    enhanced_text أيضًا للتوافق مع الواجهة الأمامية الحالية.
    """
    initial: DraftState = {
        "raw_draft": draft,
        "customer_id": customer_id,
        "language": language or detect_language(draft),
        "agent_reports": [],
        "used_llm": False,
    }
    final = _graph().invoke(initial)

    aggregated = list(final.get("grammar_changes", [])) + list(final.get("citation_changes", []))
    return {
        "refined_draft": final["final_draft"],
        "enhanced_text": final["final_draft"],      # توافق مع الواجهة الحالية
        "changes": aggregated,
        "suggestions": final.get("suggestions", []),
        "context": [{"n": a["n"], "title": a["title"], "source": a["source"], "score": a["score"]}
                    for a in final.get("articles", [])],
        "cited": final.get("cited", []),
        "decision": final.get("decision", {}),
        "agents": final.get("agent_reports", []),
        "rationale": (final.get("agent_reports") or [{}])[-1].get("rationale", ""),
        "language": final["language"],
        "source": "openai" if final.get("used_llm") else "rule_based",
    }
