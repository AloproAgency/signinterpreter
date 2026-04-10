"""V8 SignInterpreter - Database setup (SQLite + SQLAlchemy)"""
import os
from datetime import datetime, date
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from server.config import DB_PATH

engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Word(Base):
    __tablename__ = 'words'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, default='')
    created_at = Column(DateTime, default=datetime.utcnow)
    template_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    contributions = relationship('Contribution', back_populates='word', cascade='all, delete-orphan')


class Contribution(Base):
    __tablename__ = 'contributions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    contributor = Column(String, default='anonymous')
    status = Column(String, default='pending')  # pending, approved, rejected
    file_path = Column(String, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    word = relationship('Word', back_populates='contributions')


class IndexBuild(Base):
    __tablename__ = 'index_builds'
    id = Column(Integer, primary_key=True, autoincrement=True)
    built_at = Column(DateTime, default=datetime.utcnow)
    n_words = Column(Integer)
    n_templates = Column(Integer)
    summary_dim = Column(Integer)
    status = Column(String, default='success')
    duration_ms = Column(Integer)


class Member(Base):
    __tablename__ = 'members'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    tasks = relationship('MemberTask', back_populates='member', cascade='all, delete-orphan')


class DailyAssignment(Base):
    """Words assigned for a specific day. All active members must record these."""
    __tablename__ = 'daily_assignments'
    id = Column(Integer, primary_key=True, autoincrement=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    assigned_date = Column(Date, default=date.today)
    templates_required = Column(Integer, default=8)
    created_at = Column(DateTime, default=datetime.utcnow)
    word = relationship('Word')
    tasks = relationship('MemberTask', back_populates='assignment', cascade='all, delete-orphan')


class MemberTask(Base):
    """Tracks each member's progress on a daily assignment."""
    __tablename__ = 'member_tasks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(Integer, ForeignKey('members.id'), nullable=False)
    assignment_id = Column(Integer, ForeignKey('daily_assignments.id'), nullable=False)
    status = Column(String, default='pending')  # pending, done, rejected
    templates_recorded = Column(Integer, default=0)
    completed_at = Column(DateTime, nullable=True)
    rejected_reason = Column(String, nullable=True)
    member = relationship('Member', back_populates='tasks')
    assignment = relationship('DailyAssignment', back_populates='tasks')


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
