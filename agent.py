"""
agent.py
--------
هذا هو "عقل" المشروع. يبني هذا الملف رسمًا بيانيًا (Graph) باستخدام
LangGraph يمثل مسار معالجة رسالة العميل بالكامل:

    رسالة العميل
        -> تنظيف وفهم الرسالة
        -> تحديد اللغة
        -> البحث في قاعدة المعرفة (RAG)
        -> تحليل نوع المشكلة واتخاذ القرار
        -> اختيار الأداة المناسبة وتنفيذها
        -> حفظ المحادثة
        -> إعادة الرد النهائي

الوكيل يعمل في وضعين:
1. Rule-Based Mode (الافتراضي، USE_OPENAI=false): قواعد ثابتة مكتوبة
   ببايثون، تعمل فورًا دون أي مفتاح OpenAI. هذا يسمح للطالب بتجربة
   المشروع بالكامل مجانًا.
2. OpenAI Mode (USE_OPENAI=true + مفتاح صالح): يُستخدم نموذج اللغة
   لاتخاذ القرار، مع رجوع تلقائي (Fallback) لوضع القواعد في حال فشل
   الاتصال أو كان الرد غير صالح.
"""

import json
import re
from typing import Optional, TypedDict

from sqlalchemy.orm import Session

from config import settings
from models import AgentDecision
from prompts import SYSTEM_PROMPT, build_decision_prompt
import tools as tools_module
from rag import search_knowledge, RagResult

from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# كلمات مفتاحية تُستخدم في وضع القواعد (Rule-Based Mode)
# ---------------------------------------------------------------------------
_ESCALATION_KEYWORDS = [
    # عربي
    "اختراق", "اخترق", "احتيال", "احتيالي", "نصب", "خصم غير معروف",
    "خصم مالي", "مبلغ غير معروف", "شكوى", "أشتكي", "مدير", "مشرف",
    "غاضب", "غضب", "زعلان جدًا", "قانوني", "محامي", "مقاضاة", "سرقة",
    # English
    "hack", "hacked", "fraud", "scam", "unauthorized charge",
    "unknown charge", "complaint", "manager", "supervisor", "furious",
    "angry", "lawsuit", "legal action", "stolen", "steal",
]

_CATEGORY_KEYWORDS = {
    "account_access": ["كلمة المرور", "دخول", "تسجيل الدخول", "password", "login", "verification", "رمز التحقق"],
    "billing": ["فاتورة", "استرجاع", "مبلغ", "خصم", "دفع", "refund", "billing", "payment", "charge"],
    "account_update": ["بريد إلكتروني", "تحديث", "بيانات", "email", "update", "profile"],
    "order_tracking": ["طلب", "شحنة", "تتبع", "order", "tracking", "shipment"],
    "security": ["اختراق", "حماية", "أمان", "security", "hack", "breach"],
}

_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")


# ---------------------------------------------------------------------------
# حالة الرسم البياني (Graph State)
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    customer_id: str
    raw_message: str
    cleaned_message: str
    language: str

    rag_found: bool
    rag_answer: Optional[str]
    rag_source: Optional[str]
    rag_confidence: float

    ticket_reference: Optional[int]

    decision_action: str
    decision_category: str
    decision_priority: str
    decision_subject: str
    decision_reason: str
    decision_requires_human: bool
    decision_source: str  # "rule_based" أو "openai"

    tool_result: dict
    final_reply: str
    final_ticket_id: Optional[int]


# ---------------------------------------------------------------------------
# دوال مساعدة لوضع القواعد (Rule-Based)
# ---------------------------------------------------------------------------
def detect_language(text: str) -> str:
    """يحدد لغة النص بطريقة مبسطة: عربي إن وُجد حرف عربي، وإلا إنجليزي."""
    return "ar" if _ARABIC_RANGE.search(text) else "en"


def _classify_category(text: str, rag_section_title: Optional[str]) -> str:
    lowered = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lowered:
                return category
    if rag_section_title:
        return "general"
    return "general"


def _has_escalation_trigger(text: str) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in _ESCALATION_KEYWORDS)


def rule_based_decision(
    message: str,
    language: str,
    rag_result: RagResult,
    ticket_reference: Optional[int],
) -> AgentDecision:
    """
    منطق اتخاذ القرار الافتراضي (بدون OpenAI). يعتمد على:
    - وجود إشارة لرقم تذكرة -> check_ticket.
    - وجود كلمات تصعيد حساسة -> escalate.
    - وجود إجابة موثوقة من RAG -> answer.
    - غير ذلك -> create_ticket.
    """
    category = _classify_category(message, rag_result.section_title)

    # 1) هل يسأل العميل عن تذكرة موجودة؟
    if ticket_reference is not None:
        return AgentDecision(
            action="check_ticket",
            category=category,
            priority="low",
            subject="استفسار عن حالة تذكرة",
            reason="تم العثور على رقم تذكرة داخل رسالة العميل.",
            requires_human=False,
        )

    # 2) هل تحتوي الرسالة على مثيرات تصعيد حساسة؟
    if _has_escalation_trigger(message):
        return AgentDecision(
            action="escalate",
            category="security" if category == "general" else category,
            priority="urgent",
            subject="حالة حساسة تحتاج تدخلاً بشريًا فوريًا",
            reason="تم رصد كلمات تدل على اختراق/احتيال/شكوى قوية/غضب شديد.",
            requires_human=True,
        )

    # 3) هل توجد إجابة موثوقة في قاعدة المعرفة؟
    if rag_result.found:
        return AgentDecision(
            action="answer",
            category=category,
            priority="low",
            subject=rag_result.section_title or "استفسار عام",
            reason="تم العثور على إجابة موثوقة في قاعدة المعرفة.",
            requires_human=False,
        )

    # 4) لا توجد إجابة موثوقة ولا تصعيد ولا تذكرة -> افتح تذكرة عادية
    return AgentDecision(
        action="create_ticket",
        category=category,
        priority="medium",
        subject=message[:80] if message else "طلب دعم عام",
        reason="لم يتم العثور على إجابة موثوقة في قاعدة المعرفة، ويحتاج الأمر متابعة من فريق الدعم.",
        requires_human=True,
    )


def _extract_json_object(text: str) -> Optional[dict]:
    """يحاول استخراج أول كائن JSON صالح من نص قد يحتوي على زوائد."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def openai_decision(
    message: str,
    language: str,
    rag_result: RagResult,
    ticket_reference: Optional[int],
) -> tuple[AgentDecision, str]:
    """
    يحاول اتخاذ القرار عبر OpenAI. في حال أي فشل (لا يوجد مفتاح،
    خطأ اتصال، رد غير صالح JSON...) يرجع تلقائيًا لوضع القواعد.

    يعيد tuple: (القرار, المصدر الفعلي المستخدم "openai" أو "rule_based")
    """
    if not settings.use_openai or not settings.openai_api_key:
        return rule_based_decision(message, language, rag_result, ticket_reference), "rule_based"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)

        rag_context = None
        if rag_result.found:
            rag_context = f"[{rag_result.section_title}] {rag_result.answer} (confidence={rag_result.confidence})"

        user_prompt = build_decision_prompt(message, language, rag_context)

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)

        if parsed is None:
            # الرد ليس JSON صالحًا -> نرجع لوضع القواعد بدل تعطيل النظام
            return rule_based_decision(message, language, rag_result, ticket_reference), "rule_based"

        decision = AgentDecision(**parsed)
        return decision, "openai"

    except Exception:
        # أي خطأ (مفتاح خاطئ، انقطاع اتصال، Timeout...) -> رجوع آمن للقواعد
        return rule_based_decision(message, language, rag_result, ticket_reference), "rule_based"


# ---------------------------------------------------------------------------
# عُقد الرسم البياني (Graph Nodes)
# ---------------------------------------------------------------------------
def _clean_message_node(state: AgentState) -> AgentState:
    cleaned = " ".join(state["raw_message"].split())
    return {"cleaned_message": cleaned}


def _detect_language_node(state: AgentState) -> AgentState:
    if state.get("language"):
        return {}
    return {"language": detect_language(state["cleaned_message"])}


def _search_kb_node(state: AgentState) -> AgentState:
    result = search_knowledge(state["cleaned_message"])
    return {
        "rag_found": result.found,
        "rag_answer": result.answer,
        "rag_source": result.source,
        "rag_confidence": result.confidence,
    }


def _decide_node(state: AgentState) -> AgentState:
    rag_result = RagResult(
        found=state.get("rag_found", False),
        answer=state.get("rag_answer"),
        source=state.get("rag_source"),
        section_title=None,
        confidence=state.get("rag_confidence", 0.0),
    )
    ticket_reference = tools_module.extract_ticket_reference(state["cleaned_message"])

    decision, source = openai_decision(
        state["cleaned_message"], state["language"], rag_result, ticket_reference
    )

    return {
        "ticket_reference": ticket_reference,
        "decision_action": decision.action,
        "decision_category": decision.category,
        "decision_priority": decision.priority,
        "decision_subject": decision.subject,
        "decision_reason": decision.reason,
        "decision_requires_human": decision.requires_human,
        "decision_source": source,
    }


def _route_after_decision(state: AgentState) -> str:
    """يحدد إلى أي عقدة (Tool) ننتقل بناءً على قرار الوكيل."""
    action = state["decision_action"]
    if action in {"answer", "create_ticket", "escalate", "check_ticket"}:
        return action
    return "create_ticket"  # قيمة آمنة افتراضية في حال قرار غير متوقع


def _make_answer_node(db: Session):
    def _answer_node(state: AgentState) -> AgentState:
        # الإجابة موجودة أصلًا من نتيجة RAG المحسوبة سابقًا
        reply = state.get("rag_answer") or "عذرًا، لم أتمكن من إيجاد إجابة دقيقة لهذا الاستفسار."
        return {
            "tool_result": {"success": True, "used_tool": "search_knowledge_tool"},
            "final_reply": reply,
            "final_ticket_id": None,
        }

    return _answer_node


def _make_create_ticket_node(db: Session):
    def _create_ticket_node(state: AgentState) -> AgentState:
        result = tools_module.create_ticket_tool(
            db=db,
            customer_id=state["customer_id"],
            subject=state["decision_subject"],
            description=state["cleaned_message"],
            category=state["decision_category"],
            priority=state["decision_priority"],
        )
        if result["success"]:
            reply = (
                f"لم أجد إجابة جاهزة وموثوقة لطلبك، لذا قمت بفتح تذكرة دعم رقم "
                f"{result['ticket_id']} وسيتواصل معك فريق الدعم قريبًا."
                if state["language"] == "ar"
                else (
                    f"I could not find a fully reliable answer, so I opened support "
                    f"ticket #{result['ticket_id']}. Our support team will follow up soon."
                )
            )
        else:
            reply = (
                "حدث خطأ أثناء فتح التذكرة، يرجى المحاولة لاحقًا."
                if state["language"] == "ar"
                else "Something went wrong while creating your ticket. Please try again later."
            )
        return {
            "tool_result": {**result, "used_tool": "create_ticket_tool"},
            "final_reply": reply,
            "final_ticket_id": result.get("ticket_id"),
        }

    return _create_ticket_node


def _make_escalate_node(db: Session):
    def _escalate_node(state: AgentState) -> AgentState:
        result = tools_module.escalate_ticket_tool(
            db=db,
            customer_id=state["customer_id"],
            subject=state["decision_subject"],
            description=state["cleaned_message"],
            reason=state["decision_reason"],
        )
        if result["success"]:
            reply = (
                f"أتفهم مدى أهمية هذا الأمر. تم تصعيد حالتك فورًا إلى فريق الدعم "
                f"المتقدم برقم تذكرة {result['ticket_id']}، وسيتواصلون معك في أقرب وقت ممكن."
                if state["language"] == "ar"
                else (
                    f"I understand this is important. Your case has been escalated "
                    f"immediately to our advanced support team as ticket #{result['ticket_id']}."
                )
            )
        else:
            reply = (
                "حدث خطأ أثناء تصعيد حالتك، يرجى التواصل المباشر مع الدعم الفني."
                if state["language"] == "ar"
                else "An error occurred while escalating your case. Please contact support directly."
            )
        return {
            "tool_result": {**result, "used_tool": "escalate_ticket_tool"},
            "final_reply": reply,
            "final_ticket_id": result.get("ticket_id"),
        }

    return _escalate_node


def _make_check_ticket_node(db: Session):
    def _check_ticket_node(state: AgentState) -> AgentState:
        ticket_reference = state.get("ticket_reference")
        if ticket_reference is None:
            reply = (
                "لم أتمكن من التعرف على رقم التذكرة في رسالتك، هل يمكنك إرساله مرة أخرى؟"
                if state["language"] == "ar"
                else "I could not detect a ticket number in your message. Could you resend it?"
            )
            return {
                "tool_result": {"success": False, "used_tool": "get_ticket_status_tool", "error": "no_reference"},
                "final_reply": reply,
                "final_ticket_id": None,
            }

        result = tools_module.get_ticket_status_tool(db=db, ticket_id=ticket_reference)
        if result["success"] and result["found"]:
            reply = (
                f"حالة التذكرة رقم {result['ticket_id']} حاليًا: {result['status']} "
                f"(الأولوية: {result['priority']})."
                if state["language"] == "ar"
                else f"Ticket #{result['ticket_id']} status: {result['status']} (priority: {result['priority']})."
            )
        else:
            reply = (
                f"لم يتم العثور على تذكرة بالرقم {ticket_reference}. يرجى التأكد من الرقم."
                if state["language"] == "ar"
                else f"No ticket was found with number {ticket_reference}. Please double-check it."
            )
        found_ticket_id = result.get("ticket_id") if result.get("success") and result.get("found") else None
        return {
            "tool_result": {**result, "used_tool": "get_ticket_status_tool"},
            "final_reply": reply,
            "final_ticket_id": found_ticket_id,
        }

    return _check_ticket_node


def _make_save_conversation_node(db: Session):
    def _save_conversation_node(state: AgentState) -> AgentState:
        tools_module.save_conversation_tool(
            db=db,
            customer_id=state["customer_id"],
            user_message=state["raw_message"],
            agent_reply=state["final_reply"],
            action=state["decision_action"],
            source=state["decision_source"],
            language=state["language"],
            ticket_id=state.get("final_ticket_id"),
        )
        return {}

    return _save_conversation_node


def _respond_node(state: AgentState) -> AgentState:
    # عقدة شكلية فقط لإغلاق الرسم البياني بوضوح؛ الرد النهائي مُجهّز مسبقًا
    return {}


# ---------------------------------------------------------------------------
# بناء الرسم البياني الكامل
# ---------------------------------------------------------------------------
def build_graph(db: Session):
    """يبني ويجمّع (compile) رسم LangGraph لجلسة قاعدة بيانات معينة."""
    graph = StateGraph(AgentState)

    graph.add_node("clean_message", _clean_message_node)
    graph.add_node("detect_language", _detect_language_node)
    graph.add_node("search_kb", _search_kb_node)
    graph.add_node("decide", _decide_node)

    graph.add_node("answer", _make_answer_node(db))
    graph.add_node("create_ticket", _make_create_ticket_node(db))
    graph.add_node("escalate", _make_escalate_node(db))
    graph.add_node("check_ticket", _make_check_ticket_node(db))

    graph.add_node("save_conversation", _make_save_conversation_node(db))
    graph.add_node("respond", _respond_node)

    graph.set_entry_point("clean_message")
    graph.add_edge("clean_message", "detect_language")
    graph.add_edge("detect_language", "search_kb")
    graph.add_edge("search_kb", "decide")

    graph.add_conditional_edges(
        "decide",
        _route_after_decision,
        {
            "answer": "answer",
            "create_ticket": "create_ticket",
            "escalate": "escalate",
            "check_ticket": "check_ticket",
        },
    )

    for tool_node in ("answer", "create_ticket", "escalate", "check_ticket"):
        graph.add_edge(tool_node, "save_conversation")

    graph.add_edge("save_conversation", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def run_agent(db: Session, customer_id: str, message: str, language: Optional[str] = None) -> dict:
    """
    نقطة الدخول الرئيسية التي يستخدمها main.py (وأيضًا الاختبارات).

    تُشغّل الرسم البياني الكامل وتعيد Dictionary جاهزًا لملء نموذج
    ChatResponse مباشرة.
    """
    app_graph = build_graph(db)

    initial_state: AgentState = {
        "customer_id": customer_id,
        "raw_message": message,
        "language": language or "",
    }

    final_state = app_graph.invoke(initial_state)

    return {
        "action": final_state["decision_action"],
        "reply": final_state["final_reply"],
        "source": final_state.get("rag_source") if final_state["decision_action"] == "answer" else None,
        "ticket_id": final_state.get("final_ticket_id"),
        "confidence": final_state.get("rag_confidence", 0.0),
        "category": final_state.get("decision_category"),
        "priority": final_state.get("decision_priority"),
        "requires_human": final_state.get("decision_requires_human", False),
    }
