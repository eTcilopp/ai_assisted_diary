from openai import OpenAI
import os
from sqlalchemy import Column, Integer, Float, String, Date, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from models import DiaryPost, TextAnalysis, User, Comment
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from datetime import datetime

EMBEDDINGS_MODEL = "text-embedding-ada-002"
AI_MODEL = "gpt-3.5-turbo"

class Context:
    pass

class Database:
    def __init__(self, database_location: str, echo=False):
        self.engine = create_engine(database_location, echo=echo)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()


def get_diary_post(session):
        return session.query(DiaryPost).where(DiaryPost.id==2).first()

# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def get_embedding(text, client, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    
    return str(response.data[0].embedding)

def process_text(text_obj, ai_client, session):
    text_analysis = TextAnalysis()
    if isinstance(text_obj, DiaryPost):
        text_analysis.diary_post_id = text_obj.id
    else:
        text_analysis.inbound_comment_id = text_obj.id
    text_analysis.user_id = text_obj.user_id
    text_analysis.hash = hash(text_obj.text)
    text_analysis.embeddings = get_embedding(text_obj.text, ai_client)
    
    system_prompt = """
    You are a chatbot that analyses psychological state of a user by analyzing records in a personal diary.
    Responce with a JSON with 6 parameters:
    'mood' of the user when writing text in one word,
    'sentiment' of the text in one word. For example: 'positive', 'negative', 'neutral'
    'tone' of the text in one word. For example:  'anxious', 'happy', 'angry'
    'key_words' - most indicative word from the text or certain words or phrases can be indicative of mental states or issues,
    'writing_style' - writing  of the text in a few words,
    'notes' - notes about the text which could be used in the future when analyzing the dymanic of user's psychological state.
    Use English.
    """
    chat_completion = ai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text_obj.text
            }
        ],
        model=AI_MODEL,
        max_tokens=400
        
    )
    response = chat_completion.choices[0].message.content
    data = json.loads(response)
    
    text_analysis.mood = data.get('mood')
    text_analysis.sentiment = data.get('sentiment')
    text_analysis.tone = data.get('tone')
    text_analysis.indicative_words = data.get('key_words')
    text_analysis.writing_style = data.get('writing_style')
    text_analysis.notes = data.get('notes')
    
    session.add(text_analysis)
    session.commit()
    
    return text_analysis


def get_context(text_analysis, session):
    context = Context()

    latest_posts = session.query(DiaryPost).where(DiaryPost.user_id==text_analysis.user_id).where(DiaryPost.id!=text_analysis.diary_post_id).order_by(DiaryPost.id.desc()).limit(3)
    context.latest_posts_sorted = [(post.date.strftime('%Y-%m-%d'), post.text) for post in latest_posts[::-1]]
    
    analysis_history = session.query(DiaryPost, TextAnalysis).where(DiaryPost.id==TextAnalysis.diary_post_id).where(TextAnalysis.id!=text_analysis.id).where(DiaryPost.user_id==text_analysis.user_id).order_by(DiaryPost.id.desc()).limit(30)
    analysis_history = analysis_history[::-1]

    context.mood_history =  [(post.date.strftime('%Y-%m-%d'), analysis.mood) for (post, analysis) in analysis_history]
    context.semtiment_history = [(post.date.strftime('%Y-%m-%d'), analysis.sentiment) for (post, analysis) in analysis_history]
    context.tone_history = [(post.date.strftime('%Y-%m-%d'), analysis.tone) for (post, analysis) in analysis_history]
    context.indicative_words_history = [(post.date.strftime('%Y-%m-%d'), analysis.indicative_words) for (post, analysis) in analysis_history]
    context.writing_style_history = [(post.date.strftime('%Y-%m-%d'), analysis.writing_style) for (post, analysis) in analysis_history]
    context.notes_history = [(post.date.strftime('%Y-%m-%d'), analysis.notes) for (post, analysis) in analysis_history]

    return context
    
def get_ai_reply(ai_client, diary_post, text_analysis, context):
    pass
    

def run():
    db = Database("sqlite:///database.db")
    session = db.session
    ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    diary_post = get_diary_post(session)
    # text_analysis = process_text(diary_post, ai_client, session)
    text_analysis= session.query(TextAnalysis).where(TextAnalysis.id==2).first()
    context = get_context(text_analysis, session)
    ai_reply = get_ai_reply(ai_client, diary_post, text_analysis, context)
    

if __name__ == "__main__":
    run()