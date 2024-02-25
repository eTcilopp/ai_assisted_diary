from sqlalchemy import orm, create_engine, Column, Integer, String, Date, DateTime, ForeignKey, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the base class
Base = orm.declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(String)
    notes = Column(Text)
    extended_properties = Column(Text)
    # created = Column(DateTime, default=func.now())

class DiaryPost(Base):
    __tablename__ = 'diary_posts'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    date = Column(Date)
    text = Column(Text)
    extended_properties = Column(Text)

class Comment(Base):
    __tablename__ = 'comments'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    diary_post_id = Column(Integer, ForeignKey('diary_posts.id'))
    parent_comment_id = Column(Integer, ForeignKey('comments.id'))
    date = Column(Date)
    text = Column(Text)
    extended_properties = Column(Text)

class TextAnalysis(Base):
    __tablename__ = 'text_analysis'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    diary_post_id = Column(Integer, ForeignKey('diary_posts.id'))
    inbound_comment_id = Column(Integer, ForeignKey('comments.id'))
    hash = Column(String)
    embeddings = Column(Text)
    mood = Column(Text)
    sentiment = Column(String)
    tone = Column(String)
    indicative_words = Column(String)
    writing_style = Column(String)
    notes = Column(Text)
    outbound_comment_id = Column(Integer, ForeignKey('comments.id'))
    meta_data = Column(Text)
    extended_properties = Column(Text)


engine = create_engine('sqlite:///database.db')
Base.metadata.create_all(engine)

# TODO: This was not tested yet
# ai_user = User(name='AI', age='2020-01-01', notes='AI user', extended_properties='{}')
# Session = sessionmaker(bind=engine)
# session = Session()
# session.add(ai_user)
# session.commit()
# session.close()
