"""
models.py
---------
نماذج Pydantic المستخدمة في التحقق من صحة البيانات (Validation)
وفي توثيق الـ API تلقائيًا عبر Swagger.

هذه النماذج منفصلة تمامًا عن نماذج قاعدة البيانات (SQLAlchemy)
الموجودة في database.py، وهذا فصل معماري سليم بين طبقة البيانات
وطبقة التحقق/العرض.
"""

from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, Field, ConfigDict

# القيم المسموح بها لحالة التذكرة وأولويتها وفعل الوكيل
TicketStatus = Literal["open", "in_progress", "resolved", "closed"]
TicketPriority = Literal["low", "medium", "high", "urgent"]
AgentAction = Literal["answer", "create_ticket", "escalate", "check_ticket"]


# ---------------------------------------------------------------------------
# نماذج المحادثة (Chat)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """الطلب القادم من العميل عبر POST /chat"""

    customer_id: str = Field(..., min_length=1, description="معرّف العميل الفريد")
    message: str = Field(..., min_length=1, description="نص رسالة العميل")
    language: Optional[Literal["ar", "en"]] = Field(
        default=None, description="لغة الرسالة إن كانت معروفة مسبقًا"
    )


class ChatResponse(BaseModel):
    """الرد الذي يعيده الوكيل الذكي بعد اتخاذ القرار وتنفيذ الإجراء"""

    action: AgentAction
    reply: str
    source: Optional[str] = Field(
        default=None, description="مصدر الإجابة إن وُجدت (اسم الملف والقسم)"
    )
    ticket_id: Optional[int] = Field(default=None, description="رقم التذكرة إن وُجدت")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    category: Optional[str] = None
    priority: Optional[str] = None
    requires_human: bool = False


# ---------------------------------------------------------------------------
# نماذج التذاكر (Tickets)
# ---------------------------------------------------------------------------
class TicketCreate(BaseModel):
    """بيانات إنشاء تذكرة دعم جديدة"""

    customer_id: str = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    category: str = Field(default="general")
    priority: TicketPriority = Field(default="medium")


class TicketResponse(BaseModel):
    """تمثيل تذكرة كاملة تُعاد للعميل"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    customer_id: str
    subject: str
    description: str
    status: TicketStatus
    priority: TicketPriority
    category: str
    escalated: bool
    assigned_to: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TicketStatusUpdate(BaseModel):
    """طلب تحديث حالة تذكرة موجودة"""

    status: TicketStatus


# ---------------------------------------------------------------------------
# نماذج العملاء (Customers)
# ---------------------------------------------------------------------------
class CustomerCreate(BaseModel):
    """بيانات إنشاء/تسجيل عميل جديد"""

    customer_id: str = Field(..., min_length=1)
    name: Optional[str] = None
    email: Optional[str] = None
    preferred_language: Literal["ar", "en"] = "ar"


# ---------------------------------------------------------------------------
# نموذج قرار الوكيل (Agent Decision)
# ---------------------------------------------------------------------------
class AgentDecision(BaseModel):
    """
    الشكل المنظم لقرار الوكيل الذكي، سواء جاء من قواعد ثابتة
    (Rule-Based) أو من نموذج OpenAI. هذا هو "العقد" (contract)
    الذي يجب أن يلتزم به أي مصدر لاتخاذ القرار داخل المشروع.
    """

    action: AgentAction
    category: str = "general"
    priority: TicketPriority = "medium"
    subject: str = ""
    reason: str = ""
    requires_human: bool = False
