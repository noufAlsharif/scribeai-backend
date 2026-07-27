"""
rag.py
------
نظام استرجاع معزز بالبيانات (RAG) بسيط جدًا، مصمم خصيصًا ليعمل
بدون أي مكتبات ذكاء اصطناعي خارجية (بدون FAISS أو Chroma أو
Embeddings). الفكرة:

1. نقرأ ملف قاعدة المعرفة (data/knowledge_base.txt) ونقسمه إلى
   "أقسام" Sections، كل قسم يبدأ بعنوان يبدأ بـ "## ".
2. عند وصول سؤال من العميل، نحوّل نصه إلى مجموعة كلمات (tokens).
3. نقارن كلمات السؤال بكلمات كل قسم باستخدام تشابه جيب التمام
   (Cosine Similarity) المحسوب يدويًا على متجهات تكرار الكلمات
   (Term Frequency)، دون أي اعتماد على مكتبات تعلم آلي ثقيلة.
4. نعيد أفضل قسم مطابق مع درجة الثقة (confidence) الخاصة به.

هذا النهج "بسيط" كما طلب المشروع، لكنه فعّال بما يكفي لتوضيح
فكرة RAG للطلاب، ويمكن لاحقًا استبداله بـ FAISS/Chroma بسهولة لأن
الواجهة (search) لا تتغير.
"""

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from config import settings

# نمط للتعرف على الكلمات، يشمل الحروف العربية والإنجليزية والأرقام
_TOKEN_PATTERN = re.compile(r"[A-Za-z\u0600-\u06FF]+", re.UNICODE)

# كلمات شائعة لا تحمل معنى مميزًا (Stopwords) بالعربية والإنجليزية
_STOPWORDS = {
    "من", "إلى", "على", "في", "عن", "مع", "هل", "ما", "كيف", "هذا", "هذه",
    "التي", "الذي", "أن", "لا", "لم", "و", "أو", "ثم", "قد", "كان", "كل",
    "the", "a", "an", "is", "are", "to", "of", "in", "on", "and", "or",
    "for", "how", "what", "do", "does", "my", "i", "you",
}


def _tokenize(text: str) -> List[str]:
    """يحوّل نصًا حرًا إلى قائمة كلمات نظيفة (lowercase وبدون توقف)."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class KBSection:
    """يمثل قسمًا واحدًا من ملف قاعدة المعرفة."""

    title: str
    content: str
    tf: dict  # Term Frequency: عدد تكرار كل كلمة داخل القسم


@dataclass
class RagResult:
    """نتيجة البحث داخل قاعدة المعرفة."""

    found: bool
    answer: Optional[str]
    source: Optional[str]
    section_title: Optional[str]
    confidence: float


def _term_frequency(tokens: List[str]) -> dict:
    tf: dict = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1
    return tf


def _cosine_similarity(tf_a: dict, tf_b: dict) -> float:
    """يحسب تشابه جيب التمام بين متجهي تكرار كلمتين، يدويًا وبدون numpy."""
    common_keys = set(tf_a.keys()) & set(tf_b.keys())
    dot_product = sum(tf_a[k] * tf_b[k] for k in common_keys)

    norm_a = math.sqrt(sum(v * v for v in tf_a.values()))
    norm_b = math.sqrt(sum(v * v for v in tf_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _load_sections(path: str) -> List[KBSection]:
    """يقرأ ملف قاعدة المعرفة ويقسمه إلى أقسام بحسب عناوين '## '."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError as exc:
        # خطأ واضح يسهل تتبعه بدلًا من انهيار غامض لاحقًا
        raise FileNotFoundError(
            f"ملف قاعدة المعرفة غير موجود في المسار: {path}. "
            "تأكد من وجود المجلد data وملف knowledge_base.txt بداخله."
        ) from exc

    raw_sections = re.split(r"^##\s+", raw_text, flags=re.MULTILINE)
    sections: List[KBSection] = []

    for raw in raw_sections:
        raw = raw.strip()
        if not raw:
            continue
        lines = raw.splitlines()
        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        tokens = _tokenize(title + " " + content)
        sections.append(KBSection(title=title, content=content, tf=_term_frequency(tokens)))

    return sections


@lru_cache(maxsize=1)
def _get_sections_cached(path: str) -> tuple:
    """
    يخزّن الأقسام مؤقتًا (Cache) حتى لا نعيد قراءة الملف من القرص
    في كل طلب. lru_cache يتطلب مدخلات قابلة للتجزئة (hashable)
    لذلك نعيد tuple بدل list.
    """
    return tuple(_load_sections(path))


def reload_knowledge_base() -> None:
    """يمسح الذاكرة المؤقتة لإجبار النظام على إعادة قراءة الملف من القرص."""
    _get_sections_cached.cache_clear()


def search_knowledge(query: str, top_k: int = 1) -> RagResult:
    """
    الدالة الرئيسية للبحث داخل قاعدة المعرفة.

    المدخلات:
        query: رسالة العميل أو سؤاله.
        top_k: عدد النتائج المرشحة الداخلية قبل اختيار الأفضل (للتوسع لاحقًا).

    المخرجات:
        RagResult تحتوي على الإجابة والمصدر ودرجة الثقة.
        إذا كانت أفضل درجة ثقة أقل من الحد الأدنى المسموح
        (settings.min_rag_score) فإن found تكون False، ولا يُسمح
        للوكيل باستخدام هذه الإجابة.
    """
    sections = _get_sections_cached(settings.knowledge_base_path)
    query_tokens = _tokenize(query)
    query_tf = _term_frequency(query_tokens)

    if not query_tf or not sections:
        return RagResult(found=False, answer=None, source=None, section_title=None, confidence=0.0)

    scored = []
    for section in sections:
        score = _cosine_similarity(query_tf, section.tf)
        scored.append((score, section))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_section = scored[0]

    if best_score < settings.min_rag_score:
        return RagResult(
            found=False,
            answer=None,
            source=None,
            section_title=best_section.title if best_section else None,
            confidence=round(best_score, 4),
        )

    source = f"{settings.knowledge_base_path.split('/')[-1]}#{best_section.title}"
    return RagResult(
        found=True,
        answer=best_section.content,
        source=source,
        section_title=best_section.title,
        confidence=round(best_score, 4),
    )
