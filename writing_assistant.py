"""
writing_assistant.py
--------------------
مساعد الكتابة الأكاديمية لصفحة مساحة العمل — يعمل **بدون مفتاح ذكاء اصطناعي**.

يوفّر شيئين:

1) محادثة تحليل المصادر:  POST /api/workspace/chat
   يفهم أوامر مثل: لخّص / نقاط رئيسية / ماذا تقول المصادر عن ... /
   وثّق هذه الجملة / هيكل التقرير / راجع مسودتي.
   كل الإجابات مستخرجة فعليًا من مصادر التقرير — لا توليد نص وهمي.

2) تحسين أكاديمي للمسودة:  enhance_academic()
   تصحيح إملائي (بنك الكلمات) + نحو + أسلوب أكاديمي (فك الاختصارات،
   استبدال العامية، تنبيه الضمير الأول والجُمل الطويلة) + إدراج توثيق
   [1]، [2] من **مصادر التقرير نفسها** وليس من قاعدة معرفة الدعم.

مهم: لا يوجد هنا أي نبرة خدمة عملاء ولا تذاكر ولا اعتذارات — هذا الملف
مخصّص للكتابة الأكاديمية فقط.

الترقية لاحقًا إلى توليد حقيقي (اختياري ومجاني عبر Ollama):
    في .env  ->  USE_OPENAI=true
                 OPENAI_API_KEY=ollama
                 OPENAI_BASE_URL=http://localhost:11434/v1
                 OPENAI_MODEL=llama3.1
    ثم فعّل الفرع المعلّم بـ  # [LLM HOOK]  في دالة answer_writing_query.
"""

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from config import settings

# ---------------------------------------------------------------------------
# أدوات نصية (مستقلة حتى لا تعتمد على قاعدة معرفة الدعم)
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z\u0600-\u06FF0-9]+")
_STOP = {
    "من", "إلى", "على", "في", "عن", "مع", "هل", "ما", "كيف", "هذا", "هذه", "التي",
    "الذي", "أن", "لا", "لم", "و", "أو", "ثم", "قد", "كان", "كل", "the", "a", "an",
    "is", "are", "to", "of", "in", "on", "and", "or", "for", "how", "what", "do",
    "does", "my", "i", "you", "it", "this", "that", "with", "by", "can", "be",
}
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
CITE_THRESHOLD = 0.12

_WORD_BANK_PATHS = [
    getattr(settings, "word_bank_path", None),
    "data/words_alpha.txt", "words_alpha.txt", "app/data/words_alpha.txt",
]

COMMON_FIXES = {
    "colected": "collected", "helo": "hello", "teh": "the", "recieve": "receive",
    "recieved": "received", "seperate": "separate", "definately": "definitely",
    "wich": "which", "becuase": "because", "thier": "their", "adress": "address",
    "untill": "until", "begining": "beginning", "beleive": "believe",
    "acount": "account", "acounts": "accounts", "sucessful": "successful",
    "occassion": "occasion", "existance": "existence", "similiar": "similar",
    "comitted": "committed", "agumented": "augmented", "augemented": "augmented",
    "technolgy": "technology", "tecnology": "technology", "knowlege": "knowledge",
    "reserch": "research", "reasearch": "research", "anaylsis": "analysis",
    "expirement": "experiment", "hypothesys": "hypothesis", "litrature": "literature",
    "methedology": "methodology", "significiant": "significant", "diffrent": "different",
    "suports": "supports", "suport": "support", "recomend": "recommend",
    "recomended": "recommended", "developement": "development", "enviorment": "environment",
    "perfomance": "performance", "prefered": "preferred", "occured": "occurred",
    "neccessary": "necessary", "consistant": "consistent", "dependant": "dependent",
    "framwork": "framework", "libary": "library", "alogrithm": "algorithm",
    "alghorithm": "algorithm", "paramter": "parameter", "funtion": "function",
}

# أسلوب أكاديمي: فك الاختصارات واستبدال العامية بالفصيح
ACADEMIC_SUBS_EN = [
    (r"\bcan't\b", "cannot", "Expanded contraction “can't” to “cannot”"),
    (r"\bdon't\b", "do not", "Expanded contraction “don't” to “do not”"),
    (r"\bdoesn't\b", "does not", "Expanded contraction “doesn't” to “does not”"),
    (r"\bdidn't\b", "did not", "Expanded contraction “didn't” to “did not”"),
    (r"\bisn't\b", "is not", "Expanded contraction “isn't” to “is not”"),
    (r"\baren't\b", "are not", "Expanded contraction “aren't” to “are not”"),
    (r"\bwon't\b", "will not", "Expanded contraction “won't” to “will not”"),
    (r"\bit's\b", "it is", "Expanded contraction “it's” to “it is”"),
    (r"\bdoesnt\b", "does not", "Expanded contraction to “does not”"),
    (r"\ba lot of\b", "many", "Replaced informal “a lot of” with “many”"),
    (r"\blots of\b", "many", "Replaced informal “lots of” with “many”"),
    (r"\bkind of\b", "somewhat", "Replaced informal “kind of”"),
    (r"\bstuff\b", "material", "Replaced informal “stuff”"),
    (r"\bthings\b", "factors", "Replaced vague “things” with “factors”"),
    (r"\bbig\b", "substantial", "Replaced informal “big” with “substantial”"),
    (r"\bhuge\b", "considerable", "Replaced informal “huge”"),
    (r"\bgot\b", "obtained", "Replaced informal “got” with “obtained”"),
    (r"\bshow(s)? that\b", r"indicate\1 that", "Preferred “indicates that” for findings"),
]
ACADEMIC_SUBS_AR = [
    (r"\bكثير من\b", "العديد من", "صياغة أكاديمية بدل «كثير من»"),
    (r"\bأشياء\b", "عوامل", "استبدال «أشياء» بمصطلح أدقّ"),
    (r"\bكبير جدًا\b", "كبير", "حذف التهويل «جدًا»"),
]

# تنبيهات أسلوبية (لا تُستبدل تلقائيًا، تُرفع كملاحظة)
FIRST_PERSON_EN = re.compile(r"\b(I|we|my|our|me|us)\b")
FIRST_PERSON_AR = re.compile(r"\b(أنا|نحن|رأيي|أعتقد|نعتقد)\b")
LONG_SENTENCE_WORDS = 35


def tokenize(text: str) -> List[str]:
    return [w for w in _TOKEN.findall(str(text).lower()) if w not in _STOP and len(w) > 1]


def term_freq(tokens: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out


def cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def split_sentences(text: str) -> List[str]:
    out: List[str] = []
    for line in str(text).split("\n"):
        if not line.strip():
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!؟?])\s+", line) if s.strip())
    return out


# ---------------------------------------------------------------------------
# تقييم جودة الجملة
# صفحات الويب مليئة بجُمل دعائية ونداءات تسويقية لا تفيد تقريرًا أكاديميًا.
# ترجيح تكرار الكلمات وحده يختارها لأنها تكرّر اسم الموضوع. لذلك نضيف
# مقياس جودة يرفع الجُمل التعريفية/الخبرية ويخفض الدعائية.
# ---------------------------------------------------------------------------
_JUNK = re.compile(
    r"\b(click here|learn more|read more|contact us|sign ?up|subscribe|get started|"
    r"book a demo|free trial|our team|we offer|we provide|we can help|follow us|"
    r"privacy policy|cookies?|newsletter|all rights reserved|terms of service|"
    r"اتصل بنا|اشترك|سجّل الآن|تواصل معنا|جميع الحقوق|اقرأ المزيد|احصل على)\b|©",
    re.IGNORECASE)
_SECOND_PERSON = re.compile(r"\b(you|your|yours|you're|you'll|we|our|us)\b", re.IGNORECASE)
# عبارات ترويجية مبالِغة أو حشو لا يقدّم معلومة
_HYPE = re.compile(
    r"\b(truly|really|game.?chang\w+|revolution\w+|cutting.?edge|state.of.the.art|"
    r"amazing|incredible|awesome|the future of|unlock\w*|transform\w+ your|"
    r"proves? its worth|next level|seamless\w*|best.in.class|world.?class|"
    r"لا شك|بلا منازع|الأفضل على الإطلاق|ثورة حقيقية|يغيّر قواعد اللعبة)\b",
    re.IGNORECASE)
_INFORMATIVE = re.compile(
    r"\b(is|are|was|were|refers? to|defined as|known as|consists? of|comprises?|"
    r"means|includes?|involves?|based on|relies on|uses?|enables?|allows?|requires?|"
    r"according to|research|study|studies|results?|evidence|data|percent|"
    r"يُعرَّف|يُعدّ|يشير إلى|يتكون|يشمل|يعتمد|أظهرت|الدراسة|النتائج|البيانات)\b|"
    r"\d{4}|\d+\s*%",
    re.IGNORECASE)


def sentence_quality(sen: str) -> float:
    """درجة صلاحية الجملة للاستخدام الأكاديمي (سالبة = دعائية/عديمة الفائدة)."""
    text = sen.strip()
    words = text.split()
    score = 0.0
    if _JUNK.search(text):
        score -= 2.0                      # نداء تسويقي صريح
    if text.endswith(("?", "؟")):
        score -= 1.2                      # سؤال ترويجي لا يقدّم معلومة
    if text.endswith("!"):
        score -= 0.6
    score -= 0.3 * len(_SECOND_PERSON.findall(text))   # خطاب مباشر «أنت/نحن»
    if _HYPE.search(text):
        score -= 1.1                      # حشو ترويجي بلا معلومة
    if _INFORMATIVE.search(text):
        score += 0.9                      # صياغة تعريفية أو خبرية
    if len(words) < 6:
        score -= 0.8                      # عنوان أو شذرة
    elif 10 <= len(words) <= 40:
        score += 0.3                      # طول مناسب لجملة معلوماتية
    elif len(words) > 55:
        score -= 0.4
    if text.isupper():
        score -= 0.8
    if sum(c.isdigit() for c in text) > len(text) * 0.3:
        score -= 0.6                      # جدول أرقام أو كود
    return score


MIN_QUALITY = -0.5     # أقل من ذلك تُستبعد الجملة تمامًا


def rank_sentences(text: str, query_tf: Optional[Dict[str, int]] = None) -> List[tuple]:
    """
    يعيد [(score, index, sentence)] مرتّبة تنازليًا.
    الدرجة = صلة الجملة بالموضوع (تكرار الكلمات أو الاستعلام) + جودتها.
    """
    sents = split_sentences(text)
    if not sents:
        return []
    freqs = query_tf if query_tf else term_freq(tokenize(text))
    out = []
    for i, sen in enumerate(sents):
        toks = tokenize(sen)
        if not toks:
            continue
        q = sentence_quality(sen)
        if q < MIN_QUALITY:
            continue                       # استبعاد الجُمل الدعائية
        if query_tf:
            relevance = cosine(query_tf, term_freq(toks)) * 3.0
        else:
            relevance = sum(freqs.get(t, 0) for t in toks) / len(toks)
        out.append((relevance + q, i, sen))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def summarize_extractive(text: str, max_sentences: int = 4) -> str:
    """تلخيص استخراجي يستبعد الجُمل الدعائية ويفضّل التعريفية والخبرية."""
    ranked = rank_sentences(text)
    if not ranked:                          # لا جملة صالحة -> أعد أول ما يوجد
        sents = split_sentences(text)
        return " ".join(sents[:max_sentences])
    top = ranked[:max_sentences]
    return " ".join(s for _, _, s in sorted(top, key=lambda x: x[1]))


# ---------------------------------------------------------------------------
# بنك الكلمات
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_word_bank() -> frozenset:
    for path in _WORD_BANK_PATHS:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return frozenset(w.strip().lower() for w in f if w.strip())
    return frozenset()


def _edits1(word: str) -> Set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return set(
        [a + b[1:] for a, b in splits if b]
        + [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        + [a + c + b[1:] for a, b in splits if b for c in _LETTERS]
        + [a + c + b for a, b in splits for c in _LETTERS]
    )


def _known1(word: str, bank: frozenset) -> Set[str]:
    return {e for e in _edits1(word) if e in bank}


def _match_case(orig: str, fix: str) -> str:
    if len(orig) > 1 and orig.isupper():
        return fix.upper()
    if orig[:1].isupper():
        return fix[:1].upper() + fix[1:]
    return fix


def _R(lang: str, ar: str, en: str) -> str:
    return ar if lang == "ar" else en


def detect_lang(text: str) -> str:
    return "ar" if re.search(r"[\u0600-\u06FF]", str(text)) else "en"


# ===========================================================================
# مصادر التقرير: تمثيل موحّد
# ===========================================================================
def _source_text(s: dict) -> str:
    """كل النص المتاح لمصدر واحد (العنوان + الملخّص + المحتوى)."""
    return " ".join(str(s.get(k) or "") for k in ("title", "summary", "content")).strip()


def _prep_sources(sources: List[dict]) -> List[dict]:
    prepped = []
    for i, s in enumerate(sources or []):
        text = _source_text(s)
        prepped.append({
            "n": int(s.get("n") or i + 1),
            "title": (s.get("title") or f"Source {i+1}").strip(),
            "summary": (s.get("summary") or "").strip(),
            "content": (s.get("content") or "").strip(),
            "url": (s.get("url") or "").strip(),
            "ieee": (s.get("ieee_citation") or "").strip(),
            "text": text,
            "tf": term_freq(tokenize(text)),
        })
    return prepped


# ===========================================================================
# 1) محادثة مساعد الكتابة
# ===========================================================================
INTENT_PATTERNS = [
    ("summarize", r"لخ+ص|ملخ+ص|تلخيص|\bsummar(y|ise|ize)\b|\btl;?dr\b"),
    ("keypoints", r"نقاط|أهم|رئيسي|\bkey ?points?\b|\bbullets?\b|\bmain ideas?\b"),
    ("outline", r"هيكل|مخطط|خطة|أقسام التقرير|\boutline\b|\breport structure\b"),
    ("cite", r"وث+ق|توثيق|مرجع|مصدر لهذه|\bcite\b|\bcitation\b|\breference for\b"),
    # استخراج موجّه من مصدر طويل: «استخرج أنواع VR» / «what are the types of ...»
    ("extract", r"استخرج|استخرجي|أنواع|انواع|أشكال|عد+د|اذكر|\bextract\b|\btypes? of\b|"
                r"\bkinds? of\b|\bcategories\b|\blist\b|\bwhat are the\b|\bexamples? of\b"),
    ("search", r"ماذا تقول|ابحث|ما هي|دليل|\bwhat do\b|\bfind\b|\bevidence\b|\bsay about\b"),
    ("help", r"مساعدة|ماذا تستطيع|\bhelp\b|\bwhat can you\b|\bcommands?\b"),
]

# أنماط تدل على تعداد داخل النص
_LIST_LINE = re.compile(r"^\s*(?:[-*•·—–]|\(?\d+[.)]|[a-zA-Z][.)])\s+(.{3,})$", re.MULTILINE)
_ENUM_CUE = re.compile(
    r"(types?|kinds?|categor\w+|forms?|classes|varieties|include[sd]?|such as|"
    r"consists? of|divided into|classified|أنواع|أشكال|أصناف|تشمل|منها|تتضمن|تنقسم|تصنّف)",
    re.IGNORECASE)


def extract_headings(text: str, limit: int = 12) -> List[str]:
    """
    يستخرج العناوين الفرعية من نص المصدر:
      - عناوين ويكيبيديا  == العنوان ==
      - أسطر قصيرة بلا نقطة نهائية تبدو كعناوين (2..9 كلمات)
    يعيد قائمة عناوين نظيفة بلا تكرار.
    """
    heads: List[str] = []
    seen: Set[str] = set()

    def _add(h: str, strict: bool) -> None:
        h = re.sub(r"\s+", " ", h).strip(" =:#-\u2013\u2014\t")
        key = h.lower()
        if not h or key in seen:
            return
        nwords = len(h.split())
        # عناوين ويكيبيديا الصريحة: نتساهل (كلمة واحدة أو تنتهي بـ ؟ مقبولة)
        # الأسطر المخمَّنة: نتشدّد لتجنّب الجُمل
        if strict:
            if not (2 <= nwords <= 9): return
            if h.endswith((".", "\u060c", "\u061f", "?", "!", ":")): return
        else:
            if not (1 <= nwords <= 10): return
            if h.endswith((".", "\u060c")): return
        seen.add(key); heads.append(h)

    # ١) عناوين ويكيبيديا الصريحة == ... == (متساهل)
    for m in re.finditer(r"^\s*=+\s*([^=\n]{2,80}?)\s*=+\s*$", str(text), re.MULTILINE):
        _add(m.group(1), strict=False)

    # ٢) إن لم توجد، جرّب الأسطر القصيرة التي تشبه العناوين (متشدّد)
    if not heads:
        for line in str(text).split("\n"):
            ln = line.strip()
            if not ln or len(ln) > 70:
                continue
            if ln[0].isupper() or re.match(r"[\u0600-\u06FF]", ln):
                _add(ln, strict=True)
            if len(heads) >= limit:
                break

    return heads[:limit]


def split_sections(text: str) -> List[tuple]:
    """
    يقسّم النص إلى (عنوان القسم, نصه).
    يدعم عناوين ويكيبيديا  == العنوان ==  وإلا يقسّم على الفقرات.
    """
    padded = "\n" + str(text) + "\n"
    parts = re.split(r"\n=+\s*([^=\n]{2,80}?)\s*=+\n", padded)
    if len(parts) > 2:
        secs = [("", parts[0])]
        for i in range(1, len(parts) - 1, 2):
            secs.append((parts[i].strip(), parts[i + 1]))
        return [(t, b) for t, b in secs if b.strip()]
    return [("", p) for p in re.split(r"\n\s*\n", str(text)) if p.strip()]


def detect_intent(message: str) -> str:
    low = str(message).lower()
    for name, pattern in INTENT_PATTERNS:
        if re.search(pattern, low):
            return name
    # رسالة قصيرة جدًا بلا أمر واضح -> اعرض المساعدة
    if len(tokenize(low)) <= 2:
        return "help"
    return "search"      # الافتراضي: ابحث في المصادر عن سؤال المستخدم


def _strip_command(message: str) -> str:
    """يزيل كلمة الأمر ليبقى الموضوع/السؤال."""
    out = str(message)
    for _, pattern in INTENT_PATTERNS:
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip(" :،,.?؟")


# أفعال الأمر فقط — تُحذف مع الإبقاء على الكلمات الدلالية مثل «أنواع/types»
_EXTRACT_VERBS = re.compile(
    r"^\s*(استخرج(?:ي)?|اذكر|عد+د|أعطني|اعطني|extract|list|show me|give me|tell me)\s*[:：]?\s*",
    re.IGNORECASE)


def _strip_extract_verb(message: str) -> str:
    out = _EXTRACT_VERBS.sub("", str(message))
    return re.sub(r"\s+", " ", out).strip(" :،,.?؟")


# جسر ثنائي اللغة لمصطلحات شائعة: يسمح بسؤال عربي عن مصدر إنجليزي والعكس
_BILINGUAL = {
    "أنواع": ["types", "kinds", "categories"],
    "انواع": ["types", "kinds", "categories"],
    "أشكال": ["forms", "types"],
    "أصناف": ["categories", "classes"],
    "تطبيقات": ["applications", "uses"],
    "استخدامات": ["uses", "applications"],
    "فوائد": ["benefits", "advantages"],
    "مزايا": ["advantages", "benefits"],
    "تحديات": ["challenges", "limitations"],
    "قيود": ["limitations", "constraints"],
    "عيوب": ["disadvantages", "drawbacks"],
    "تاريخ": ["history"],
    "أجهزة": ["hardware", "devices", "headsets"],
    "عتاد": ["hardware", "devices"],
    "صحة": ["health", "effects"],
    "تعريف": ["definition"],
    "أمثلة": ["examples"],
    "مكونات": ["components"],
    "طرق": ["methods", "techniques"],
    "منهجية": ["methodology", "methods"],
    "نتائج": ["results", "findings"],
    "مستقبل": ["future"],
    "أمن": ["security"],
    "تعليم": ["education", "learning"],
}
_BILINGUAL_REV = {}
for _ar, _ens in _BILINGUAL.items():
    for _en in _ens:
        _BILINGUAL_REV.setdefault(_en, []).append(_ar)


def _ar_stem(tok: str) -> str:
    """يزيل أداة التعريف وبعض السوابق ليطابق القاموس («التطبيقات» -> «تطبيقات»)."""
    for pre in ("وال", "بال", "كال", "فال", "ال"):
        if tok.startswith(pre) and len(tok) - len(pre) >= 3:
            return tok[len(pre):]
    return tok


def _expand_query(query: str) -> Dict[str, int]:
    """
    يبني متجّه الاستعلام مع إضافة المقابلات بالعربية/الإنجليزية للمصطلحات
    الشائعة، حتى يعمل سؤال عربي على مصدر إنجليزي.
    """
    toks = tokenize(query)
    extra: List[str] = []
    for tok in toks:
        stem = _ar_stem(tok)
        if stem != tok:
            extra.append(stem)                 # أضف الجذر نفسه للمطابقة العربية
        for key in {tok, stem}:
            extra.extend(_BILINGUAL.get(key, []))
            extra.extend(_BILINGUAL_REV.get(key, []))
    return term_freq(toks + extra)


def _dedup_key(text: str) -> str:
    """مفتاح موحّد للتخلّص من التكرار بين عناصر القوائم والجُمل."""
    t = re.sub(r"^\s*(?:[-*•·—–]|\(?\d+[.)]|[a-zA-Z][.)])\s*", "", str(text))
    t = re.sub(r"[^\w\u0600-\u06FF]+", " ", t.lower()).strip()
    return t[:60]


def _help_text(lang: str, srcs: List[dict]) -> str:
    if lang == "ar":
        return (
            "أستطيع مساعدتك في كتابة التقرير اعتمادًا على مصادرك فقط (بدون اختراع محتوى):\n"
            "• «لخّص» — ملخّص للمصدر المحدّد\n"
            "• «نقاط رئيسية» — أهم النقاط في صيغة قائمة\n"
            "• «ماذا تقول المصادر عن الأمن السيبراني؟» — يبحث في كل المصادر ويعيد الجُمل المطابقة مع أرقام التوثيق\n"
            "• «وثّق: <الجملة>» — يقترح المصدر الأنسب ورقمه\n"
            "• «هيكل التقرير» — مخطط أقسام مبني على عناوين مصادرك\n"
            f"\nالمصادر المتاحة حاليًا: {len(srcs)}"
        )
    return (
        "I can help you write the report using only your own sources (nothing invented):\n"
        "• “summarize” — a summary of the selected source\n"
        "• “key points” — the main points as a list\n"
        "• “what do the sources say about reinforcement learning?” — searches every source and returns matching sentences with citation numbers\n"
        "• “cite: <your sentence>” — suggests the best-matching source and its number\n"
        "• “outline” — a section plan built from your source titles\n"
        f"\nSources currently available: {len(srcs)}"
    )


def _no_sources(lang: str) -> dict:
    return {
        "reply": _R(lang,
                    "لا توجد مصادر في هذا التقرير بعد. أضف مصدرًا من شاشة «المصادر» ثم اسألني عنه.",
                    "This report has no sources yet. Add one from the Sources screen, then ask me about it."),
        "intent": "help", "citations": [], "insert": "", "source": "rule_based",
    }


def answer_writing_query(
    message: str,
    sources: List[dict],
    draft: str = "",
    active_source_n: Optional[int] = None,
    language: Optional[str] = None,
) -> dict:
    """
    يعيد dict فيه: reply, insert (نص جاهز للإدراج), citations, intent, source.

    # [LLM HOOK] عند توفّر نموذج (OpenAI أو Ollama محلي) يمكن استبدال هذه
    # الدالة بنداء واحد يمرّر: نص المصادر + المسودة + رسالة المستخدم.
    """
    lang = language or detect_lang(message)
    srcs = _prep_sources(sources)
    intent = detect_intent(message)

    if intent == "help":
        return {"reply": _help_text(lang, srcs), "intent": "help",
                "citations": [], "insert": "", "source": "rule_based"}

    if not srcs and intent in ("summarize", "keypoints", "search", "cite", "outline"):
        return _no_sources(lang)

    active = next((s for s in srcs if s["n"] == active_source_n), None) or (srcs[0] if srcs else None)

    # ---------------- تلخيص ----------------
    if intent == "summarize":
        topic = _strip_command(message)
        target = active
        if topic:                       # «لخّص <عنوان مصدر>»
            t_tf = term_freq(tokenize(topic))
            best, best_score = None, 0.0
            for s in srcs:
                sc = cosine(t_tf, s["tf"])
                if sc > best_score:
                    best_score, best = sc, s
            if best and best_score >= 0.15:
                target = best
        body = target["content"] or target["summary"] or target["text"]
        summary = summarize_extractive(body, 3)
        if not summary:
            return {"reply": _R(lang, "هذا المصدر لا يحتوي نصًا كافيًا للتلخيص.",
                                "This source has no extractable text to summarize."),
                    "intent": intent, "citations": [], "insert": "", "source": "rule_based"}
        reply = _R(lang, f"ملخّص «{target['title']}» [{target['n']}]:\n{summary}",
                        f"Summary of “{target['title']}” [{target['n']}]:\n{summary}")
        return {"reply": reply, "intent": intent, "citations": [target["n"]],
                "insert": f"{summary} [{target['n']}]", "source": "rule_based"}

    # ---------------- نقاط رئيسية ----------------
    if intent == "keypoints":
        body = active["content"] or active["summary"] or active["text"]
        ranked = rank_sentences(body)[:5]
        points = [s for _, _, s in sorted(ranked, key=lambda x: x[1])]
        if not points:
            return {"reply": _R(lang, "لا يوجد نص كافٍ لاستخراج نقاط.",
                                "Not enough text to extract points."),
                    "intent": intent, "citations": [], "insert": "", "source": "rule_based"}
        bullets = "\n".join(f"• {p}" for p in points)
        head = _R(lang, f"أهم النقاط في «{active['title']}» [{active['n']}]:",
                       f"Key points from “{active['title']}” [{active['n']}]:")
        return {"reply": f"{head}\n{bullets}", "intent": intent,
                "citations": [active["n"]],
                "insert": "\n".join(f"{p} [{active['n']}]" for p in points),
                "source": "rule_based"}

    # ---------------- بحث في كل المصادر ----------------
    if intent == "search":
        query = _strip_command(message) or message
        q_tf = _expand_query(query)
        hits = []
        for s in srcs:
            for sen in split_sentences(s["content"] or s["summary"] or s["text"]):
                sc = cosine(q_tf, term_freq(tokenize(sen)))
                if sc > 0:
                    hits.append((sc, s["n"], s["title"], sen))
        hits.sort(key=lambda x: x[0], reverse=True)
        hits = [h for h in hits if h[0] >= 0.10][:4]
        if not hits:
            return {"reply": _R(lang,
                        f"لم أجد في مصادرك ما يتعلق بـ «{query}». جرّب كلمات أقرب لنص المصدر، أو أضف مصدرًا يغطّي الموضوع.",
                        f"I couldn't find anything about “{query}” in your sources. Try wording closer to the source text, or add a source covering it."),
                    "intent": intent, "citations": [], "insert": "", "source": "rule_based"}
        lines = [f"• {sen} [{n}]" for _, n, _, sen in hits]
        head = _R(lang, f"ما تقوله مصادرك عن «{query}»:", f"What your sources say about “{query}”:")
        return {"reply": f"{head}\n" + "\n".join(lines), "intent": intent,
                "citations": sorted({n for _, n, _, _ in hits}),
                "insert": " ".join(f"{sen} [{n}]" for _, n, _, sen in hits),
                "source": "rule_based"}

    # ---------------- استخراج موجّه من مصدر طويل ----------------
    if intent == "extract":
        query = _strip_extract_verb(message) or message
        q_tf = _expand_query(query)

        # ابحث في كل المصادر عن أفضل الأقسام المطابقة للطلب
        scored_secs = []
        for s in srcs:
            body = s["content"] or s["summary"] or s["text"]
            for title, sec in split_sections(body):
                blob = f"{title} {sec}"
                sc = cosine(q_tf, term_freq(tokenize(blob)))
                if title and cosine(q_tf, term_freq(tokenize(title))) > 0:
                    sc += 0.25            # ترجيح تطابق عنوان القسم
                if sc > 0:
                    scored_secs.append((sc, s, title, sec))
        scored_secs.sort(key=lambda x: x[0], reverse=True)
        top_secs = [x for x in scored_secs if x[0] >= 0.08][:2]

        if not top_secs:
            return {"reply": _R(lang,
                        f"لم أجد في مصادرك قسمًا يخصّ «{query}». تأكد أن المصدر يغطّي الموضوع، أو استخدم كلمات موجودة في نصّه.",
                        f"I couldn't find a section about “{query}” in your sources. Check that the source covers it, or use wording that appears in its text."),
                    "intent": intent, "citations": [], "insert": "", "source": "rule_based"}

        items: List[tuple] = []          # (نص العنصر, رقم المصدر)
        seen_items: Set[str] = set()
        for _, s, title, sec in top_secs:
            # ١) عناصر مكتوبة أصلًا كقائمة
            for m in _LIST_LINE.finditer(sec):
                item = re.sub(r"\s+", " ", m.group(1)).strip()
                k = _dedup_key(item)
                if 3 < len(item) < 400 and k and k not in seen_items:
                    seen_items.add(k); items.append((item, s["n"]))
            # ٢) جُمل فيها دلالة تعداد أو كلمات الطلب
            for sen in split_sentences(sec):
                clean = re.sub(r"^\s*(?:[-*•·—–]|\(?\d+[.)])\s*", "", re.sub(r"\s+", " ", sen)).strip()
                k = _dedup_key(clean)
                if not k or k in seen_items or len(clean.split()) < 4:
                    continue
                relevant = cosine(q_tf, term_freq(tokenize(clean)))
                if _ENUM_CUE.search(clean) or relevant >= 0.18:
                    seen_items.add(k)
                    items.append((clean, s["n"]))
            if len(items) >= 12:
                break

        if not items:
            # لا تعداد واضح -> أعِد القسم ملخّصًا بدل لا شيء
            _, s, title, sec = top_secs[0]
            summary = summarize_extractive(sec, 4)
            head = _R(lang, f"لم أجد تعدادًا صريحًا، لكن هذا أقرب قسم" + (f" («{title}»)" if title else "") + f" [{s['n']}]:",
                            f"No explicit list found, but here is the closest section" + (f" (“{title}”)" if title else "") + f" [{s['n']}]:")
            return {"reply": f"{head}\n{summary}", "intent": intent,
                    "citations": [s["n"]], "insert": f"{summary} [{s['n']}]", "source": "rule_based"}

        items = items[:10]
        sec_titles = [t for _, _, t, _ in top_secs if t]
        head = _R(lang,
            f"استخرجتُ عن «{query}»" + (f" من قسم «{sec_titles[0]}»" if sec_titles else "") + ":",
            f"Extracted about “{query}”" + (f" from section “{sec_titles[0]}”" if sec_titles else "") + ":")
        bullets = "\n".join(f"• {txt} [{n}]" for txt, n in items)
        return {"reply": f"{head}\n{bullets}", "intent": intent,
                "citations": sorted({n for _, n in items}),
                "insert": bullets, "source": "rule_based"}

    # ---------------- اقتراح توثيق ----------------
    if intent == "cite":
        sentence = _strip_command(message)
        if not sentence:
            return {"reply": _R(lang, "اكتب: «وثّق: الجملة التي تريد توثيقها».",
                                "Write: “cite: the sentence you want a source for”."),
                    "intent": intent, "citations": [], "insert": "", "source": "rule_based"}
        s_tf = term_freq(tokenize(sentence))
        ranked = sorted(((cosine(s_tf, s["tf"]), s) for s in srcs), key=lambda x: x[0], reverse=True)
        best_score, best = ranked[0] if ranked else (0.0, None)
        if not best or best_score < CITE_THRESHOLD:
            return {"reply": _R(lang,
                        "لا يوجد مصدر يدعم هذه الجملة بدرجة كافية. إمّا أن تعيد صياغتها لتطابق مصادرك، أو تضيف مصدرًا يدعمها — ولا تضع توثيقًا غير مطابق.",
                        "No source supports this sentence strongly enough. Either rephrase it to match your sources or add a source that supports it — don't attach a citation that doesn't fit."),
                    "intent": intent, "citations": [], "insert": "", "source": "rule_based"}
        cited = re.sub(r"([.!؟?]?)\s*$", rf" [{best['n']}]\1", sentence.strip())
        reply = _R(lang,
            f"أنسب مصدر: «{best['title']}» → [{best['n']}] (تشابه {best_score:.2f})\n{cited}",
            f"Best match: “{best['title']}” → [{best['n']}] (similarity {best_score:.2f})\n{cited}")
        if best["ieee"]:
            reply += f"\n\n[{best['n']}] {re.sub(r'^\s*\[\d+\]\s*', '', best['ieee'])}"
        return {"reply": reply, "intent": intent, "citations": [best["n"]],
                "insert": cited, "source": "rule_based"}

    # ---------------- هيكل التقرير ----------------
    if intent == "outline":
        secs_en = ["Introduction", "Background and related work", "Methods / approach",
                   "Findings and discussion", "Conclusion"]
        secs_ar = ["المقدمة", "الخلفية النظرية والأعمال السابقة", "المنهجية",
                   "النتائج والمناقشة", "الخاتمة"]
        secs = secs_ar if lang == "ar" else secs_en
        lines = []
        for i, sec in enumerate(secs, 1):
            lines.append(f"{i}. {sec}")
            if i == 2 and srcs:            # قسم الخلفية: كل مصدر مع عناوينه الفرعية
                for s in srcs:
                    lines.append(f"   {s['n']}. {s['title']} [{s['n']}]")
                    for sub in extract_headings(s["content"] or s["summary"] or s["text"])[:6]:
                        lines.append(f"      - {sub} [{s['n']}]")
        head = _R(lang, "مخطط مقترح مع العناوين الفرعية من مصادرك:",
                       "Suggested outline with the subheadings from your sources:")
        note = _R(lang, "\n(الأقسام الرئيسية هيكل تنظيمي؛ والعناوين الفرعية مستخرجة من مصادرك فعلًا.)",
                       "\n(The main sections are a scaffold; the subheadings are pulled from your actual sources.)")
        body = "\n".join(lines)
        return {"reply": f"{head}\n{body}{note}", "intent": intent,
                "citations": [s["n"] for s in srcs], "insert": body, "source": "rule_based"}


# ===========================================================================
# 2) تحسين أكاديمي للمسودة
# ===========================================================================
def _tidy(text: str) -> str:
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\s+([,.;:!?؟،])", r"\1", t)
    t = re.sub(r"([,;:،])(?=\S)", r"\1 ", t)
    t = re.sub(r"([.!?؟])(?=[A-Za-z\u0600-\u06FF])", r"\1 ", t)
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return t.strip()


def enhance_academic(draft: str, sources: List[dict], language: Optional[str] = None) -> dict:
    """
    تحسين أكاديمي: إملاء + نحو + أسلوب أكاديمي + توثيق من مصادر التقرير.
    لا نبرة خدمة عملاء ولا اعتذارات ولا خطوات تالية.
    """
    lang = language or detect_lang(draft)
    bank = load_word_bank()
    srcs = _prep_sources(sources)
    changes: List[dict] = []
    seen: Set[tuple] = set()

    # 1) إملاء
    def _record(w: str, to: str) -> None:
        if (w, to) not in seen:
            seen.add((w, to))
            changes.append({"type": "typo", "from": w, "to": to})

    def _fix(m: re.Match) -> str:
        w = m.group(0)
        low = w.lower()
        if low in COMMON_FIXES:
            to = _match_case(w, COMMON_FIXES[low]); _record(w, to); return to
        if not bank or len(low) <= 2 or low in bank:
            return w
        cands = _known1(low, bank)
        if len(cands) == 1:
            to = _match_case(w, next(iter(cands))); _record(w, to); return to
        return w

    spelled = re.sub(r"[A-Za-z]+", _fix, draft)

    # 2) نحو/ترقيم
    tidied = _tidy(spelled)
    if tidied != spelled:
        changes.append({"type": "grammar", "detail": _R(lang,
            "تصحيح المسافات وعلامات الترقيم وبدايات الجُمل",
            "Corrected spacing, punctuation & sentence casing")})

    # 3) أسلوب أكاديمي
    styled = tidied
    for pattern, repl, label in (ACADEMIC_SUBS_AR if lang == "ar" else ACADEMIC_SUBS_EN):
        new = re.sub(pattern, repl, styled, flags=re.IGNORECASE)
        if new != styled:
            changes.append({"type": "style", "detail": label})
            styled = new

    # تنبيهات لا تُطبَّق آليًا
    fp = (FIRST_PERSON_AR if lang == "ar" else FIRST_PERSON_EN).findall(styled)
    if fp:
        changes.append({"type": "note", "detail": _R(lang,
            f"ضمير المتكلّم مستخدم {len(fp)} مرة — راجع مدى ملاءمته للأسلوب الأكاديمي.",
            f"First person appears {len(fp)} time(s) — consider an impersonal construction.")})
    for s in split_sentences(styled):
        if len(s.split()) > LONG_SENTENCE_WORDS:
            changes.append({"type": "note", "detail": _R(lang,
                f"جملة طويلة ({len(s.split())} كلمة) — يُستحسن تقسيمها.",
                f"Long sentence ({len(s.split())} words) — consider splitting it.")})
            break

    # 4) توثيق من مصادر التقرير
    cited: Set[int] = set()
    out_lines: List[str] = []
    for line in styled.split("\n"):
        if not line.strip():
            out_lines.append(line); continue
        new_sens = []
        for sen in re.split(r"(?<=[.!؟?])\s+", line):
            if not sen.strip() or re.search(r"\[\d+\]", sen) or not srcs:
                new_sens.append(sen); continue
            s_tf = term_freq(tokenize(sen))
            best, best_score = None, 0.0
            for s in srcs:
                sc = cosine(s_tf, s["tf"])
                if sc > best_score:
                    best_score, best = sc, s
            if best and best_score >= CITE_THRESHOLD:
                cited.add(best["n"])
                changes.append({"type": "citation", "n": best["n"], "detail": best["title"]})
                if re.search(r"[.!؟?][\"']?\s*$", sen):
                    sen = re.sub(r"([.!؟?][\"']?)\s*$", rf" [{best['n']}]\1", sen)
                else:
                    sen = f"{sen} [{best['n']}]"
            new_sens.append(sen)
        out_lines.append(" ".join(new_sens))
    final = "\n".join(out_lines)

    if cited:
        head = "المراجع" if lang == "ar" else "References"
        by_n = {s["n"]: s for s in srcs}
        refs = []
        for n in sorted(cited):
            s = by_n[n]
            ieee = re.sub(r"^\s*\[\d+\]\s*", "", s["ieee"]) if s["ieee"] else s["title"]
            refs.append(f"[{n}] {ieee}")
        final = f"{final}\n\n{head}\n" + "\n".join(refs)

    n_typo = sum(1 for c in changes if c["type"] == "typo")
    n_style = sum(1 for c in changes if c["type"] == "style")
    n_cite = sum(1 for c in changes if c["type"] == "citation")
    rationale = _R(lang,
        f"تصحيح أكاديمي: {n_typo} إملاء، {n_style} أسلوب، {n_cite} توثيق من {len(srcs)} مصدرًا.",
        f"Academic pass: {n_typo} spelling, {n_style} style, {n_cite} citation(s) from {len(srcs)} source(s).")

    return {
        "refined_draft": final,
        "enhanced_text": final,
        "changes": changes,
        "suggestions": [],
        "context": [{"n": s["n"], "title": s["title"], "source": s["url"], "score": 0.0} for s in srcs],
        "cited": sorted(cited),
        "decision": {},
        "agents": [{"agent": "academic_editor", "rationale": rationale, "changes": changes}],
        "rationale": rationale,
        "language": lang,
        "source": "rule_based",
    }
