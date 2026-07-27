"""
database.py
-----------
إعداد قاعدة البيانات (SQLite) باستخدام SQLAlchemy، وتعريف الجداول التالية:

1. customers      -> بيانات العملاء.
2. tickets        -> تذاكر الدعم الفني.
3. conversations  -> سجل كل رسالة بين العميل والوكيل الذكي.

يتم إنشاء الجداول تلقائيًا عند أول تشغيل للتطبيق عبر init_db().
"""

from datetime import datetime
from typing import Generator

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config import settings

# -------------------------------------------------------------------
# إعداد المحرك (Engine) والجلسة (Session)
# -------------------------------------------------------------------
# check_same_thread=False مطلوب فقط مع SQLite لأن FastAPI قد يستخدم
# أكثر من Thread للتعامل مع الطلبات.
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# -------------------------------------------------------------------
# جدول العملاء
# -------------------------------------------------------------------
class Customer(Base):
    """يمثل عميلًا مسجلاً في النظام."""

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=True)
    email = Column(String(128), nullable=True)
    preferred_language = Column(String(8), default="ar")
    created_at = Column(DateTime, default=datetime.utcnow)


# -------------------------------------------------------------------
# جدول تذاكر الدعم
# -------------------------------------------------------------------
class Ticket(Base):
    """يمثل تذكرة دعم فني واحدة."""

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), index=True, nullable=False)
    subject = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)

    # open | in_progress | resolved | closed
    status = Column(String(32), default="open", index=True)

    # low | medium | high | urgent
    priority = Column(String(16), default="medium")

    category = Column(String(64), default="general")
    escalated = Column(Boolean, default=False)
    assigned_to = Column(String(64), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# -------------------------------------------------------------------
# جدول المحادثات
# -------------------------------------------------------------------
class Conversation(Base):
    """يمثل رسالة واحدة ضمن محادثة بين العميل والوكيل الذكي."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), index=True, nullable=False)
    user_message = Column(Text, nullable=False)
    agent_reply = Column(Text, nullable=False)

    # answer | create_ticket | escalate | check_ticket
    action = Column(String(32), nullable=False)

    # rule_based | openai
    source = Column(String(32), default="rule_based")

    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=True)
    language = Column(String(8), default="ar")

    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """
    ينشئ جميع الجداول في قاعدة البيانات إذا لم تكن موجودة.
    تُستدعى هذه الدالة مرة واحدة عند إقلاع التطبيق (startup event).
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency خاصة بـ FastAPI تفتح جلسة قاعدة بيانات لكل طلب
    وتغلقها تلقائيًا بعد الانتهاء، حتى في حال حدوث خطأ.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_customer(db: Session, customer_id: str, language: str = "ar") -> Customer:
    """
    يبحث عن عميل موجود بواسطة customer_id، وإذا لم يكن موجودًا
    ينشئ سجلًا جديدًا له تلقائيًا. تُستخدم هذه الدالة من داخل
    الأدوات (tools) ومسار المحادثة الرئيسي.
    """
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if customer is None:
        customer = Customer(customer_id=customer_id, preferred_language=language)
        db.add(customer)
        db.commit()
        db.refresh(customer)
    return customer
