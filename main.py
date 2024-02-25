from openai import OpenAI
import os
import ast
from typing import List, Tuple
from scipy import spatial
import numpy as np
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Date,
    DateTime,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from models import DiaryPost, TextAnalysis, User, Comment
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from datetime import datetime

EMBEDDINGS_MODEL = "text-embedding-ada-002"
AI_MODEL = "gpt-3.5-turbo"
# AI_MODEL = "gpt-4"

# TODO: need to add token count
# TODO: Improve prompts
# TODO: Need to add processing of commentsas well (and hostory of the comments)


class Context:
    pass


class Database:
    def __init__(self, database_location: str, echo=False):
        self.engine = create_engine(database_location, echo=echo)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()


def get_diary_post(session):
    return session.query(DiaryPost).where(DiaryPost.id == 5).first()


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def get_similar_post(text_analysis, session, number_of_posts=3, exclude_post_ids=[]):
    query_embedding = list(ast.literal_eval(text_analysis.embeddings.strip('[]')))
    embeddings_str = (
        session.query(TextAnalysis)
        .where(TextAnalysis.diary_post_id != text_analysis.diary_post_id)
        .where(TextAnalysis.user_id == text_analysis.user_id)
        .where(~TextAnalysis.diary_post_id.in_(exclude_post_ids))
    ).with_entities(TextAnalysis.diary_post_id, TextAnalysis.embeddings).all()
    embeddings = [list(ast.literal_eval(embedding[1].strip('[]'))) for embedding in embeddings_str]

    distances = distances_from_embeddings(query_embedding, embeddings)
    indices_of_nearest_neighbors = np.argsort(distances)[:number_of_posts]
    closest_comment_ids = [embeddings_str[i][0] for i in indices_of_nearest_neighbors]
    return session.query(DiaryPost).where(DiaryPost.id.in_(closest_comment_ids)).all()




# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def get_embedding(text, client, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)

    return str(response.data[0].embedding)


def get_ai_completion(ai_client: OpenAI, messages: list):
    chat_completion = ai_client.chat.completions.create(
        messages=messages,
        model=AI_MODEL,
        max_tokens=800,
    )
    return chat_completion.choices[0].message.content


def process_text(text_obj, ai_client, session):
    text_analysis = TextAnalysis()
    # add useful information about user to the user table
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
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_obj.text},
    ]
    response = get_ai_completion(ai_client, messages)
    data = json.loads(response)

    text_analysis.mood = data.get("mood")
    text_analysis.sentiment = data.get("sentiment")
    text_analysis.tone = data.get("tone")
    text_analysis.indicative_words = data.get("key_words")
    text_analysis.writing_style = data.get("writing_style")
    text_analysis.notes = data.get("notes")

    session.add(text_analysis)
    session.commit()

    return text_analysis


def get_context(text_analysis, session, number_of_posts=3):
    context = Context()
    # TODO: Do next! Add AI replies to the context
    user = session.query(User).where(User.id == text_analysis.user_id).first()
    context.user_name = user.name
    context.user_age = user.age

    latest_posts = (
        session.query(DiaryPost)
        .where(DiaryPost.user_id == text_analysis.user_id)
        .where(DiaryPost.id != text_analysis.diary_post_id)
        .order_by(DiaryPost.id.desc())
        .limit(number_of_posts).all()
    )
    context.latest_posts_sorted = [
        (post.date.strftime("%Y-%m-%d"), post.text) for post in latest_posts[::-1]
    ]

    context.similar_posts = get_similar_post(
        text_analysis,
        session,
        number_of_posts=3,
        exclude_post_ids=[post.id for post in latest_posts])

    analysis_history = (
        session.query(DiaryPost, TextAnalysis)
        .where(DiaryPost.id == TextAnalysis.diary_post_id)
        .where(TextAnalysis.id != text_analysis.id)
        .where(DiaryPost.user_id == text_analysis.user_id)
        .order_by(DiaryPost.id.desc())
        .limit(30)
    )
    analysis_history = analysis_history[::-1]

    context.mood_history = [
        (post.date.strftime("%Y-%m-%d"), analysis.mood)
        for (post, analysis) in analysis_history
    ]
    context.semtiment_history = [
        (post.date.strftime("%Y-%m-%d"), analysis.sentiment)
        for (post, analysis) in analysis_history
    ]
    context.tone_history = [
        (post.date.strftime("%Y-%m-%d"), analysis.tone)
        for (post, analysis) in analysis_history
    ]
    context.indicative_words_history = [
        (post.date.strftime("%Y-%m-%d"), analysis.indicative_words)
        for (post, analysis) in analysis_history
    ]
    context.writing_style_history = [
        (post.date.strftime("%Y-%m-%d"), analysis.writing_style)
        for (post, analysis) in analysis_history
    ]
    context.notes_history = [
        (post.date.strftime("%Y-%m-%d"), analysis.notes)
        for (post, analysis) in analysis_history
    ]

    return context


def get_ai_reply(ai_client, diary_post, text_analysis, context, session):
    system_prompt = f"""
    You are a professional psychologist. You are analyzing a diary post of a patient. You have the following information:
    Patient's Name: {context.user_name};
    Patient's Age: {context.user_age};
    A Few Latest post: {context.latest_posts_sorted};
    A Few Similar Posts: {context.similar_posts};
    Mood History: {context.mood_history};
    Sentiment History: {context.semtiment_history};
    Tone History: {context.tone_history};
    Indicative Words History: {context.indicative_words_history};
    Writing Style History: {context.writing_style_history};
    Your Notes History: {context.notes_history};
    Now, prepare reply to the following diary post of the patient: {diary_post.text}
    You have the following information about this post:
    Mood: {text_analysis.mood};
    Sentiment: {text_analysis.sentiment};
    Tone: {text_analysis.tone};
    Indicative Words: {text_analysis.indicative_words};
    Writing Style: {text_analysis.writing_style};
    Your Notes: {text_analysis.notes};
    You goal is to improve the psychological state of the patient. You can ask questions, give advice, or just provide support.
    Use same language as the patient used in the diary post.
    """
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    response = get_ai_completion(ai_client, messages)

    system_prompt = f"""
    Review and improve if needed the reply of a professional psychologist to the diary post of the patient.
    Make the reply less formal and more human like. Provide only improved text.
    `{response}`
    """

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    response = get_ai_completion(ai_client, messages)

    comment = Comment()
    comment.user_id = 1
    comment.diary_post_id = diary_post.id
    comment.date = datetime.now()
    comment.text = response
    comment.text_analysis_id = text_analysis.id
    session.add(comment)
    session.commit()


def run():
    db = Database("sqlite:///database.db")
    session = db.session
    ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    diary_post = get_diary_post(session)
    # text_analysis = process_text(diary_post, ai_client, session)
    text_analysis = session.query(TextAnalysis).where(TextAnalysis.id == 3).first()
    context = get_context(text_analysis, session)
    ai_reply = get_ai_reply(ai_client, diary_post, text_analysis, context, session)


if __name__ == "__main__":
    run()
