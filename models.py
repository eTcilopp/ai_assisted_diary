from sqlalchemy import (orm, create_engine,
                        Column, Integer, Float, String,
                        Date, DateTime, ForeignKey, Text,
                        func)
import enum
import _osx_support
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# Define the base class
Base = orm.declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    external_id = Column(Integer, unique=True)
    name = Column(String(100))
    email = Column(String(100))
    age = Column(String(100))
    notes = Column(Text)
    extended_properties = Column(Text)
    # created = Column(DateTime, default=func.now())


class DiaryPost(Base):
    __tablename__ = 'diary_posts'
    PROCESSED = 'processed'
    UNPROCESSED = 'unprocessed'
    IGNORED = 'ignored'
    REJECTED = 'rejected'
    id = Column(Integer, primary_key=True)
    external_id = Column(Integer, unique=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    date = Column(Date)
    text = Column(Text)
    extended_properties = Column(Text)
    text_analysis_id = Column(Integer, ForeignKey('text_analysis.id'))
    status = Column(Text, default=UNPROCESSED, nullable=True)


class Comment(Base):
    __tablename__ = 'comments'
    PROCESSED = 'processed'
    UNPROCESSED = 'unprocessed'
    IGNORED = 'ignored'
    REJECTED = 'rejected'
    id = Column(Integer, primary_key=True)
    external_id = Column(Integer, unique=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    diary_post_id = Column(Integer, ForeignKey('diary_posts.id'))
    parent_comment_id = Column(Integer, ForeignKey('comments.id'))
    date = Column(Date)
    text = Column(Text)
    extended_properties = Column(Text)
    text_analysis_id = Column(Integer, ForeignKey('text_analysis.id'))
    status = Column(Text, default=UNPROCESSED, nullable=True)


class TextAnalysis(Base):
    __tablename__ = 'text_analysis'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    diary_post_id = Column(Integer, ForeignKey('diary_posts.id'))
    inbound_comment_id = Column(Integer, ForeignKey('comments.id'))
    hash = Column(String(100))
    embeddings = Column(Text)
    mood = Column(Text)
    sentiment = Column(String(100))
    tone = Column(String(100))
    indicative_words = Column(String(500))
    writing_style = Column(String(100))
    notes = Column(Text)
    outbound_comment_id = Column(Integer, ForeignKey('comments.id'))
    meta_data = Column(Text)
    extended_properties = Column(Text)


class ContextType(Base):
    __tablename__ = 'context_types'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    extended_properties = Column(Text)


class AiModel(Base):
    __tablename__ = 'ai_models'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    extended_properties = Column(Text)


class TokenUsage(Base):
    __tablename__ = 'token_usage'
    id = Column(Integer, primary_key=True)
    created = Column(Integer)
    user_id = Column(Integer, ForeignKey('users.id'))
    context_type_id = Column(Integer, ForeignKey('context_types.id'))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    completion_tokens = Column(Integer)
    prompt_tokens = Column(Integer)
    total_tokens = Column(Integer)
    ai_model_id = Column(Integer, ForeignKey('ai_models.id'))
    extended_properties = Column(Text)


class TimeUsage(Base):
    __tablename__ = 'time_usage'
    id = Column(Integer, primary_key=True)
    created = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey('users.id'))
    context_type_id = Column(Integer, ForeignKey('context_types.id'))
    context_object_id = Column(Integer)
    elapsed = Column(Float)
    extended_properties = Column(Text)


# engine = create_engine('sqlite:///database.db')
# Base.metadata.create_all(engine)

# TODO: This was not tested yet
# ai_user = User(name='AI', age='2020-01-01', notes='AI user', extended_properties='{}')
# Session = sessionmaker(bind=engine)
# session = Session()
# session.add(ai_user)
# session.commit()
# session.close()
