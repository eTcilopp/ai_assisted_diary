from openai import OpenAI
import os
import ast
from typing import List, Tuple
from scipy import spatial
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
from models import DiaryPost, TextAnalysis, User, Comment, TokenUsage, TimeUsage, ContextType, AiModel
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from datetime import datetime
import api_calls
import sched
import time
from api_calls import get_external_ai_user_id, get_latest_posts_from_diary, get_new_external_users, get_latest_comments_from_diary, publish_response_to_diary

EMBEDDINGS_MODEL = "text-embedding-ada-002"
AI_MODEL = "gpt-3.5-turbo"
# AI_MODEL = "gpt-4"

# TODO: Clean up formatting


class Context:
    pass


class Database:
    def __init__(self, database_location: str, echo=False):
        self.engine = create_engine(database_location, echo=echo)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()


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
def get_embedding(user_id, text_obj, client, model=EMBEDDINGS_MODEL):
    text = text_obj.text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    register_token_usage(user_id, text_obj, response.usage, response.model)

    return str(response.data[0].embedding)


def get_ai_completion(user_id: int, context_obj, ai_client: OpenAI, messages: list):
    chat_completion = ai_client.chat.completions.create(
        messages=messages,
        model=AI_MODEL,
        max_tokens=800,
    )
    register_token_usage(user_id, context_obj, chat_completion.usage, chat_completion.model)
    return chat_completion.choices[0].message.content


def process_text(text_obj, ai_client):
    text_analysis = TextAnalysis()
    # TODO: add useful information about user to the user table
    if isinstance(text_obj, DiaryPost):
        text_analysis.diary_post_id = text_obj.id
    else:
        text_analysis.inbound_comment_id = text_obj.id
    
    user_id = text_obj.user_id
    text_analysis.user_id = user_id

    system_prompt = """
You are an advanced AI chatbot designed to analyze the psychological state of a user by examining entries from a personal diary.
If provided text is meaningless, too short or contains no relevant information, please respond with JSON: {"rejected": True}.
Otherwise proceed.
Your analysis should be comprehensive, focusing on subtle cues and patterns in the writing to assess the user's emotional and mental state accurately. Your response should be structured as a JSON object with the following six parameters:

- 'mood': Describe the user's mood at the time of writing in a single word, based on the overall emotional tone of the entry.
- 'sentiment': Categorize the overall sentiment of the text in one word, such as 'positive', 'negative', or 'neutral', taking into account the nuances and context of the writing.
- 'tone': Identify the tone of the text in one word, such as 'anxious', 'happy', 'angry', etc., considering both the explicit and implicit emotional expressions.
- 'key_words': List the most indicative words or phrases from the text that signal specific mental states, emotional conditions, or recurring themes, highlighting their significance in understanding the user's psychological state.
- 'writing_style': Describe the style of writing in a few words, focusing on aspects such as verbosity, coherence, formality, and any stylistic devices used that may give insights into the user's state of mind.
- 'notes': Provide observations or notes about the diary entry that could be pivotal in analyzing the dynamics of the user's psychological state over time. This may include changes in writing style, frequency of certain moods or tones, or emerging patterns that warrant attention.

Your analysis should be sensitive to the complexities of human emotion and psychological states, using the provided data to offer nuanced insights into the user's wellbeing.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_obj.text},
    ]
    response = get_ai_completion(user_id, text_obj, ai_client, messages)
    data = json.loads(response)

    if data.get("rejected") is True:
        text_analysis.rejected = True
        return text_analysis

    text_analysis.hash = hash(text_obj.text)
    text_analysis.embeddings = get_embedding(user_id, text_obj, ai_client)
    text_analysis.mood = data.get("mood")
    text_analysis.sentiment = data.get("sentiment")
    text_analysis.tone = data.get("tone")
    text_analysis.indicative_words = str(data.get("key_words")).strip('[]')
    text_analysis.writing_style = data.get("writing_style")
    text_analysis.notes = data.get("notes")

    session.add(text_analysis)
    session.commit()

    return text_analysis


def retrieve_ai_comment(post_id: int, session):
    comment = session.query(Comment).where(Comment.diary_post_id == post_id).where(Comment.user_id == AI_USER_ID).order_by(Comment.date.desc()).first()
    return comment.text if comment else None


def get_comment_history(inbound_comment_id):
    # comment_id = text_analysis.inbound_comment_id
    history = list()
    comment_obj = session.query(Comment).where(Comment.id == inbound_comment_id).first()
    while comment_obj:
        history.insert(0, (comment_obj.user_id, comment_obj.text))
        diary_post_id = comment_obj.diary_post_id
        comment_obj = session.query(Comment).where(Comment.id == comment_obj.parent_comment_id).first()
    diary_post = session.query(DiaryPost).where(DiaryPost.id == diary_post_id).first()

    messages = [{"role": "user", "content": diary_post.text}]
    for record in history:
        if record[0] == AI_USER_ID:
            messages.append({"role": "assistant", "content": record[1]})
        else:
            messages.append({"role": "user", "content": record[1]})

    return messages


def get_comment_context(text_analysis):
    context = Context()
    user = session.query(User).where(User.id == text_analysis.user_id).first()
    context.user_name = user.name
    context.user_age = user.age
    context.messages = get_comment_history(text_analysis.inbound_comment_id)

    return context


def get_post_context(text_analysis, number_of_posts=3):
    context = Context()
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
        (post.date.strftime("%Y-%m-%d"), post.text, retrieve_ai_comment(post.id, session)) for post in latest_posts[::-1]
    ]

    similar_posts = get_similar_post(
        text_analysis,
        session,
        number_of_posts=3,
        exclude_post_ids=[post.id for post in latest_posts])
    context.similar_posts = [
        (post.date.strftime("%Y-%m-%d"), post.text, retrieve_ai_comment(post.id, session)) for post in similar_posts
    ]

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
    context.sentiment_history = [
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


def arrange_posts_to_string(posts: List[Tuple[str, str, str]], headers: Tuple) -> str:
    res = ''
    for post in posts:
        res += f'{headers[0]}:\n{post[0]}\n{headers[1]}:\n{post[1]}\n'
        try:
            if post[2]:
                res += f'{headers[2]}:\n{post[2]}\n-------------------\n'
        except IndexError:
            pass
    return res


def get_post_context_type_id(context_obj):
    context_type = session.query(ContextType).\
        where(ContextType.name == context_obj.__class__.__name__).first()
    if not context_type:
        context_type = ContextType()
        context_type.name = context_obj.__class__.__name__
        session.add(context_type)
        session.commit()
    return context_type.id


def get_ai_model_id(ai_model_name):
    ai_model = session.query(AiModel).where(AiModel.name == ai_model_name).first()
    if not ai_model:
        ai_model = AiModel()
        ai_model.name = ai_model_name
        session.add(ai_model)
        session.commit()
    return ai_model.id


def register_time_usage(user_id, text_object, start_time):
    time_usage = TimeUsage()

    time_usage.user_id = user_id
    time_usage.context_type_id = get_post_context_type_id(text_object)
    time_usage.context_object_id = text_object.id
    time_usage.elapsed = (datetime.now() - start_time).total_seconds()

    session.add(time_usage)
    session.commit()


def register_token_usage(user_id, context_obj, usage, model_name):
    token_usage = TokenUsage()

    token_usage.user_id = user_id
    token_usage.context_type_id = get_post_context_type_id(context_obj)
    token_usage.context_object_id = context_obj.id

    token_usage.completion_tokens = getattr(usage, 'completion_tokens', None)
    token_usage.prompt_tokens = usage.prompt_tokens
    token_usage.total_tokens = usage.total_tokens
    token_usage.ai_model_id = get_ai_model_id(model_name)

    session.add(token_usage)
    session.commit()


def get_ai_reply_to_comment(user_id, ai_client, text_obj, text_analysis, context):
    system_prompt = f"""
 As a professional psychologist, you are analyzing a recent comment from a patient to understand and potentially improve their psychological state. Below is the relevant information extracted from the patient's comment and your analysis:

 - Patient's Information:
    - Name: {context.user_name}
    - Age: {context.user_age}
    Current Comment Analysis (this is the comment you are replying to):
    - Mood: {text_analysis.mood}
    - Sentiment: {text_analysis.sentiment}
    - Tone: {text_analysis.tone}
    - Indicative Words: {text_analysis.indicative_words}
    - Writing Style: {text_analysis.writing_style}
    - Your Notes: {text_analysis.notes}
    """
    messages = [
        {"role": "system", "content": system_prompt}
    ] + context.messages

    response = get_ai_completion(user_id, text_obj, ai_client, messages)

    # comment = Comment()
    # comment.user_id = AI_USER_ID
    # comment.parent_comment_id = text_obj.id
    # comment.date = datetime.now()
    # comment.text = response
    # comment.text_analysis_id = text_analysis.id
    # session.add(comment)
    # session.commit()

    # return comment.text
    return response


def get_ai_reply_to_post(
    user_id,
    ai_client,
    diary_post,
    text_analysis,
    context
):
    system_prompt = f"""
As a professional psychologist, you are analyzing a recent diary post from a patient to understand and potentially improve their psychological state. Below is the relevant information extracted from the patient's diary and your analysis:

- Patient's Information:
    - Name: {context.user_name}
    - Age: {context.user_age}

- Diary Analysis Context:
    - Latest Posts: {arrange_posts_to_string(context.latest_posts_sorted, ('Date', 'Post', "Psychologist's Reply"))}
    - Similar Posts: {arrange_posts_to_string(context.similar_posts, ('Date', 'Post', "Psychologist's Reply"))}
    - Mood History: {arrange_posts_to_string(context.mood_history, ('Date', 'Mood'))}
    - Sentiment History: {arrange_posts_to_string(context.sentiment_history, ('Date', 'Sentiment'))}
    - Tone History: {arrange_posts_to_string(context.tone_history, ('Date', 'Tone'))}
    - Indicative Words History: {arrange_posts_to_string(context.indicative_words_history, ('Date', 'Indicative Words'))}
    - Writing Style History: {arrange_posts_to_string(context.writing_style_history, ('Date', 'Writing Style'))}
    - Psychologist's Notes History: {arrange_posts_to_string(context.notes_history, ('Date', 'Psychologist Notes'))}

- Current Diary Post Analysis:
    - Text: {diary_post.text}
    - Mood: {text_analysis.mood}
    - Sentiment: {text_analysis.sentiment}
    - Tone: {text_analysis.tone}
    - Indicative Words: {text_analysis.indicative_words}
    - Writing Style: {text_analysis.writing_style}
    - Your Notes: {text_analysis.notes}

Your goal is to use this information to craft a response that could help improve the patient's psychological state.
In your reply, consider asking questions, offering advice, or providing support.
Ensure your response matches language style and tone used by the patient in their diary post.
Always use patient's name `{context.user_name}` in the reply - translate it if needed.
Use same language as patient does.
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    response = get_ai_completion(user_id, diary_post, ai_client, messages)

    system_prompt = f"""
Please refine the psychologist's response to the patient's diary post.
The goal is to make the communication feel more personal, relatable, and empathetic,
closely mirroring the language and tone the patient used. Focus on crafting a reply that feels like
it's coming from a caring friend rather than a distant professional.
Ensure the language is accessible, warm, and supportive, effectively bridging any emotional distance.
Adjust the text to enhance its human touch, paying special attention to nuances in the patient's original language style.
Importantly, adapt your reply to use the same language and expressions as the patient,
ensuring your response resonates more deeply and personally with them.
Patient is aware that the reply is AI generated. Please use appropiate signature at the end - like
"Warm regards, Your AI Psychologist"
or
"Best wishes, Your Digital Therapist"
or whatever fits best.
Use same language as patient does.

`{response}`
    """

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    response = get_ai_completion(user_id, diary_post, ai_client, messages)

    # comment = Comment()
    # comment.user_id = AI_USER_ID
    # comment.diary_post_id = diary_post.id
    # comment.date = datetime.now()
    # comment.text = response
    # comment.text_analysis_id = text_analysis.id
    # comment.status = Comment.IGNORED
    # session.add(comment)
    # session.commit()
    
    # return comment.text
    return response


def get_ai_user_id():
    external_user_id = get_external_ai_user_id(session)
    internal_ai_user = session.query(User).where(User.email == os.environ.get("AI_USER_EMAIL")).first()
    if not internal_ai_user:
        internal_ai_user = User(name=os.environ.get("AI_USER_NAME"), email=os.environ.get("AI_USER_EMAIL"), external_id=external_user_id, age=0)
        session.add(internal_ai_user)
        session.commit()

    return internal_ai_user.id


def update_users():
    latest_user_id = session.query(User).order_by(User.id.desc()).first().id
    new_external_users = get_new_external_users(latest_user_id)
    for user in new_external_users:
        user_obj = User()
        user_obj.external_id = user['external_id']
        user_obj.name = user['name']
        user_obj.email = user['email']
        session.add(user_obj)
        session.commit()


def update_diary_posts():
    last_internal_post = session.query(DiaryPost).order_by(DiaryPost.id.desc()).first()
    last_internal_post_id = getattr(last_internal_post, 'id', 0)
    new_external_posts = get_latest_posts_from_diary(last_internal_post_id)
    for post in new_external_posts:
        internal_diary_post = DiaryPost()
        internal_diary_post.external_id = post['external_id']
        internal_diary_post.user_id = session.query(User).filter(User.external_id == post['author_id']).first().id
        internal_diary_post.date = datetime.strptime(post['created'], '%a, %d %b %Y %H:%M:%S %Z')
        internal_diary_post.text = post['text']
        session.add(internal_diary_post)
    session.commit()


def update_comments():
    latest_internal_comment = session.query(Comment).order_by(Comment.id.desc()).first()
    last_external_comment_id = getattr(latest_internal_comment, 'external_id', 0)
    new_external_comments = get_latest_comments_from_diary(last_external_comment_id)
    for external_comment in new_external_comments:
        internal_comment = Comment()
        internal_comment.external_id = external_comment['external_id']
        internal_comment.user_id = session.query(User).filter(User.external_id == external_comment['author_id']).first().id
        internal_comment.date = datetime.strptime(external_comment['created'], '%a, %d %b %Y %H:%M:%S %Z')
        if external_comment['parent_post_id']:
            internal_comment.diary_post_id = session.query(DiaryPost).filter(DiaryPost.external_id == external_comment['parent_post_id']).first().id
        else:
            internal_comment.parent_comment_id = session.query(Comment).filter(Comment.external_id == external_comment['parent_comment_id']).first().id
        internal_comment.text = external_comment['text']
        if external_comment['author_id'] == session.query(User).filter(User.external_id == external_comment['author_id']).first().id:
            internal_comment.status = Comment.IGNORED
        session.add(internal_comment)
        session.commit()


def get_post_to_process():
    post_to_process_lst = []

    posts = session.query(DiaryPost).where(DiaryPost.status == 'unprocessed').all()

    user_to_post_dict = {}

    for post in posts:
        if post.user_id not in user_to_post_dict:
            user_to_post_dict[post.user_id] = []
        user_to_post_dict[post.user_id].append(post)

    for user_id, user_posts in user_to_post_dict.items():
        user_selected_post = sorted(user_posts, key=lambda o: o.id)[-1]
        post_to_process_lst.append(user_selected_post)

        for ignored_post in user_posts:
            if ignored_post.id != user_selected_post.id:
                ignored_post.status = DiaryPost.IGNORED
                session.add(ignored_post)

    session.commit()
    return post_to_process_lst


def get_comments_to_process():
    comments_to_process_lst = []

    comments = session.query(Comment).\
        filter(Comment.status == 'unprocessed').\
        filter(
            Comment.parent_comment_id.in_(session.query(Comment.id).filter(Comment.user_id == AI_USER_ID))
        ).all()

    user_to_comment_dict = {}

    for comment in comments:
        if comment.user_id not in user_to_comment_dict:
            user_to_comment_dict[comment.user_id] = []
        user_to_comment_dict[comment.user_id].append(comment)

    for user_id, user_comments in user_to_comment_dict.items():
        user_selected_comment = sorted(user_comments, key=lambda o: o.id)[-1]
        comments_to_process_lst.append(user_selected_comment)

        for ignored_comment in user_comments:
            if ignored_comment.id != user_selected_comment.id:
                ignored_comment.status = Comment.IGNORED
                session.add(ignored_comment)

    session.commit()
    return comments_to_process_lst


def set_record_status(record, status):
    record.status = status
    session.add(record)
    session.commit()


def publish_ai_reply(record_to_reply_to, ai_reply):
    if isinstance(record_to_reply_to, DiaryPost):
        parent_post_id = record_to_reply_to.external_id
        parent_comment_id = None
    else:
        parent_post_id = None
        parent_comment_id = record_to_reply_to.external_id

    publish_response_to_diary(
        external_parent_post_id=parent_post_id,
        external_parent_comment_id=parent_comment_id,
        text=ai_reply,
        ai_user_id=AI_USER_ID
    )


def run():
    print(f"{datetime.now()} Running AI processing...")
    db = Database("sqlite:///database.db")
    ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    global session
    session = db.session
    global AI_USER_ID
    AI_USER_ID = get_ai_user_id()
    update_users()
    update_diary_posts()
    update_comments()

    posts_to_process = get_post_to_process()
    comments_to_process = get_comments_to_process()
    for record in comments_to_process + posts_to_process:
        user_id, text_obj = record.user_id, record
        start_time = datetime.now()
        text_analysis = process_text(text_obj, ai_client)
        if getattr(text_analysis, 'rejected', False):
            register_time_usage(user_id, text_obj, start_time)
            set_record_status(record, 'rejected')
            continue
        if isinstance(text_obj, DiaryPost):
            context = get_post_context(text_analysis)
            ai_reply = get_ai_reply_to_post(user_id, ai_client, text_obj, text_analysis, context)
        else:
            context = get_comment_context(text_analysis)
            ai_reply = get_ai_reply_to_comment(user_id, ai_client, text_obj, text_analysis, context)

        set_record_status(record, 'processed')
        publish_ai_reply(text_obj, ai_reply)

        register_time_usage(user_id, text_obj, start_time)


def main():
    scheduler = sched.scheduler(time.time, time.sleep)
    interval = (60 * 60 * 6)

    def periodic_task(sc):
        run()
        sc.enter(interval, 1, periodic_task, (sc,))

    # Schedule the initial task
    scheduler.enter(interval, 1, periodic_task, (scheduler,))
    # Run the scheduler
    scheduler.run()


if __name__ == "__main__":
    main()
    # TODO: Need to add commot try/except(?)
    # TODO; Move system prompts to a separate file
