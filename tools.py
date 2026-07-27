"""
tools.py
--------
هذا الملف يحتوي على "الأدوات" (Tools) التي يستطيع الوكيل الذكي
استدعاءها لتنفيذ إجراء فعلي، بدلًا من الاكتفاء بالرد النصي فقط.

كل أداة هنا:
- لها Docstring واضح يشرح وظيفتها.
- تستقبل مدخلات محددة (customer_id, subject, ...).
- تعيد Dictionary منظم دائمًا بنفس الشكل العام:
    {"success": bool, ...بيانات إضافية..., "error": Optional[str]}
- تتعامل مع الأخطاء المحتملة بدلًا من ترك الاستثناء ينفجر بلا داعٍ.
- تستخدم قاعدة البيانات (Session) عند الحاجة.

يمكن لاحقًا للطلاب تحويل هذه الدوال إلى "Tools" رسمية لمكتبة
LangChain/LangGraph أو OpenAI Function Calling بسهولة، لأن كل
دالة هنا مستقلة وواضحة المدخلات والمخرجات.
"""

import re
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from database import Ticket, Conversation, get_or_create_customer
import rag as rag_module

# نمط للتعرف على رقم تذكرة داخل نص حر، مثل: "TCK-12"، "#12"، "تذكرة 12"
_TICKET_REF_PATTERN = re.compile(
    r"(?:TCK-|#|تذكرة\s*رقم\s*|تذكرة\s*)(\d+)", re.IGNORECASE
)


def extract_ticket_reference(text: str) -> Optional[int]:
    """
    لماذا تحتاجها الوكيل؟
        لتحديد ما إذا كانت رسالة العميل تشير إلى رقم تذكرة موجودة
        (مثل "ما حالة التذكرة TCK-3؟") حتى يقرر استخدام check_ticket.

    كيف تعمل؟
        تبحث عن أنماط شائعة (TCK-<رقم>, #<رقم>, "تذكرة <رقم>") وتعيد
        أول رقم تجده كعدد صحيح، أو None إن لم تجد شيئًا.
    """
    match = _TICKET_REF_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def search_knowledge_tool(query: str) -> dict:
    """
    ماذا تفعل الأداة؟
        تبحث داخل قاعدة المعرفة المحلية (RAG بسيط) عن إجابة مناسبة
        لسؤال العميل.

    لماذا يحتاجها الوكيل؟
        هي الخطوة الأولى قبل اتخاذ أي قرار: إذا وُجدت إجابة موثوقة
        فلا داعٍ لإنشاء تذكرة أو إزعاج موظف بشري.

    كيف يمكن تطويرها لاحقًا؟
        استبدال البحث النصي البسيط بـ Embeddings حقيقية عبر
        FAISS أو Chroma، مع الحفاظ على نفس شكل المخرجات.
    """
    try:
        result = rag_module.search_knowledge(query)
        return {
            "success": True,
            "found": result.found,
            "answer": result.answer,
            "source": result.source,
            "section_title": result.section_title,
            "confidence": result.confidence,
            "error": None,
        }
    except FileNotFoundError as exc:
        return {
            "success": False,
            "found": False,
            "answer": None,
            "source": None,
            "section_title": None,
            "confidence": 0.0,
            "error": str(exc),
        }


def create_ticket_tool(
    db: Session,
    customer_id: str,
    subject: str,
    description: str,
    category: str = "general",
    priority: str = "medium",
) -> dict:
    """
    ماذا تفعل الأداة؟
        تنشئ تذكرة دعم فني جديدة في قاعدة البيانات وتربطها بالعميل
        (وتنشئ سجل العميل تلقائيًا إذا لم يكن موجودًا).

    لماذا يحتاجها الوكيل؟
        عندما لا تكون هناك إجابة موثوقة في قاعدة المعرفة، يجب على
        الوكيل ألا "يخترع" حلاً، بل يفتح تذكرة تُحال لاحقًا لموظف.

    كيف يمكن تطويرها لاحقًا؟
        إرسال إشعار فعلي (بريد/واتساب) لفريق الدعم عند إنشاء التذكرة،
        أو توزيع التذاكر تلقائيًا على الموظفين حسب التخصص.
    """
    try:
        get_or_create_customer(db, customer_id)

        ticket = Ticket(
            customer_id=customer_id,
            subject=subject.strip() or "طلب دعم عام",
            description=description.strip(),
            category=category,
            priority=priority,
            status="open",
            escalated=False,
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)

        return {
            "success": True,
            "ticket_id": ticket.id,
            "status": ticket.status,
            "priority": ticket.priority,
            "error": None,
        }
    except Exception as exc:  # حماية عامة من أي خطأ غير متوقع في قاعدة البيانات
        db.rollback()
        return {"success": False, "ticket_id": None, "status": None, "priority": None, "error": str(exc)}


def get_ticket_status_tool(db: Session, ticket_id: int) -> dict:
    """
    ماذا تفعل الأداة؟
        تجلب حالة تذكرة موجودة برقمها.

    لماذا يحتاجها الوكيل؟
        لتنفيذ فعل "check_ticket" عندما يسأل العميل عن تذكرة سابقة.

    كيف يمكن تطويرها لاحقًا؟
        إعادة سجل كامل لتاريخ تحديثات التذكرة (Audit Log) وليس فقط
        الحالة الحالية.
    """
    try:
        ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if ticket is None:
            return {
                "success": False,
                "found": False,
                "ticket_id": ticket_id,
                "error": f"لا توجد تذكرة بالرقم {ticket_id}.",
            }
        return {
            "success": True,
            "found": True,
            "ticket_id": ticket.id,
            "subject": ticket.subject,
            "status": ticket.status,
            "priority": ticket.priority,
            "category": ticket.category,
            "escalated": ticket.escalated,
            "assigned_to": ticket.assigned_to,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
            "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
            "error": None,
        }
    except Exception as exc:
        return {"success": False, "found": False, "ticket_id": ticket_id, "error": str(exc)}


def update_ticket_status_tool(db: Session, ticket_id: int, new_status: str) -> dict:
    """
    ماذا تفعل الأداة؟
        تحدّث حالة تذكرة موجودة (open, in_progress, resolved, closed).

    لماذا يحتاجها الوكيل؟
        تُستخدم غالبًا من قِبل الموظف البشري عبر الـ API مباشرة، وليس
        من الوكيل نفسه، لكنها موجودة هنا لتكتمل دورة حياة التذكرة.

    كيف يمكن تطويرها لاحقًا؟
        إشعار العميل تلقائيًا عند تغيّر حالة تذكرته.
    """
    valid_statuses = {"open", "in_progress", "resolved", "closed"}
    if new_status not in valid_statuses:
        return {"success": False, "error": f"حالة غير صحيحة: {new_status}"}

    try:
        ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if ticket is None:
            return {"success": False, "error": f"لا توجد تذكرة بالرقم {ticket_id}."}

        ticket.status = new_status
        ticket.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(ticket)

        return {"success": True, "ticket_id": ticket.id, "status": ticket.status, "error": None}
    except Exception as exc:
        db.rollback()
        return {"success": False, "error": str(exc)}


def escalate_ticket_tool(
    db: Session,
    customer_id: str,
    subject: str,
    description: str,
    reason: str = "",
) -> dict:
    """
    ماذا تفعل الأداة؟
        تنشئ (أو تحدّث) تذكرة وتصعّدها فورًا بأولوية عاجلة، وتُعلّم
        الحقل escalated=True لتمييزها عن التذاكر العادية.

    لماذا يحتاجها الوكيل؟
        الحالات الحساسة (اختراق حساب، احتيال، غضب شديد...) يجب ألا
        تنتظر دورة معالجة التذاكر العادية، بل تصل مباشرة لموظف بشري.

    كيف يمكن تطويرها لاحقًا؟
        إرسال تنبيه فوري (Push/SMS) لمشرف الفريق فور التصعيد. حاليًا
        نكتفي بمحاكاة ذلك عبر حقل escalated في قاعدة البيانات فقط،
        دون إرسال أي إشعار حقيقي.
    """
    try:
        get_or_create_customer(db, customer_id)

        ticket = Ticket(
            customer_id=customer_id,
            subject=subject.strip() or "حالة مصعّدة تحتاج تدخلاً بشريًا",
            description=(description or reason).strip(),
            category="escalation",
            priority="urgent",
            status="open",
            escalated=True,
            assigned_to="فريق الدعم المتقدم",
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)

        return {
            "success": True,
            "ticket_id": ticket.id,
            "escalated": True,
            "assigned_to": ticket.assigned_to,
            "error": None,
        }
    except Exception as exc:
        db.rollback()
        return {"success": False, "ticket_id": None, "escalated": False, "error": str(exc)}


def save_conversation_tool(
    db: Session,
    customer_id: str,
    user_message: str,
    agent_reply: str,
    action: str,
    source: str,
    language: str,
    ticket_id: Optional[int] = None,
) -> dict:
    """
    ماذا تفعل الأداة؟
        تحفظ سجل رسالة العميل ورد الوكيل في جدول conversations، حتى
        يمكن الرجوع لتاريخ المحادثات لاحقًا (get_customer_history_tool).

    لماذا يحتاجها الوكيل؟
        بدون هذه الأداة تُفقد كل محادثة بعد انتهاء الطلب، بينما هي
        ضرورية للتدقيق ولتلخيص المشكلة قبل تحويلها لموظف بشري.

    كيف يمكن تطويرها لاحقًا؟
        فهرسة المحادثات القديمة داخل RAG نفسه لتحسين إجابات مستقبلية.
    """
    try:
        get_or_create_customer(db, customer_id, language=language)

        conversation = Conversation(
            customer_id=customer_id,
            user_message=user_message,
            agent_reply=agent_reply,
            action=action,
            source=source,
            ticket_id=ticket_id,
            language=language,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return {"success": True, "conversation_id": conversation.id, "error": None}
    except Exception as exc:
        db.rollback()
        return {"success": False, "conversation_id": None, "error": str(exc)}


def get_customer_history_tool(db: Session, customer_id: str, limit: int = 20) -> dict:
    """
    ماذا تفعل الأداة؟
        تجلب آخر محادثات عميل معيّن، مرتبة من الأحدث إلى الأقدم.

    لماذا يحتاجها الوكيل؟
        تساعد في "تلخيص المشكلة قبل إرسالها إلى موظف الدعم"، حيث
        يمكن قراءة آخر رسائل العميل لفهم سياق المشكلة كاملاً.

    كيف يمكن تطويرها لاحقًا؟
        تلخيص تلقائي لتاريخ المحادثة باستخدام نموذج لغوي قبل عرضه
        للموظف، بدل عرض الرسائل الخام فقط.
    """
    try:
        rows = (
            db.query(Conversation)
            .filter(Conversation.customer_id == customer_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .all()
        )
        history = [
            {
                "id": row.id,
                "user_message": row.user_message,
                "agent_reply": row.agent_reply,
                "action": row.action,
                "ticket_id": row.ticket_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
        return {"success": True, "customer_id": customer_id, "history": history, "error": None}
    except Exception as exc:
        return {"success": False, "customer_id": customer_id, "history": [], "error": str(exc)}
