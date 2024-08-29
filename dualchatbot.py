from huggingface_hub import login
import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from PIL import Image
import random
import re




# Load environment variables from the key.env file
load_dotenv(dotenv_path='key.env')
api_key = os.getenv('MY_API_KEY')
    
# Authenticate with Hugging Face
login(api_key)


# This info's at the top of each HuggingFace model page
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
#hf_model="meta-llama/Meta-Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(repo_id = hf_model)

    
# embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "data/manu"
folder_arsenal = "data/arsenal"

embeddings_stat = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)
embeddings_arsenal = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=folder_arsenal)


vector_db_arsenal = FAISS.load_local("data/arsenal/faiss_index", embeddings_arsenal, allow_dangerous_deserialization=True)    
vector_db_stat = FAISS.load_local("data/manu/faiss_index", embeddings_stat, allow_dangerous_deserialization=True)    


retriever_arsenal = vector_db_arsenal.as_retriever(search_kwargs={"k": 4})

retriever = vector_db_stat.as_retriever(search_kwargs={"k": 4})




memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
memory2 = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

template_man_utd = """You are a witty Manchester United fan chatbot. Always claim Man United is better, using specific facts and achievements. Respond to Arsenal's claims with counter-arguments. Limit your response to 2-3 sentences max.

Remember these guidelines:
1. Use specific arguments (e.g., exact trophy counts, memorable matches, legendary players).
2. Don't repeat arguments you've already used.
3. Directly counter the specific point made by the Arsenal fan.
4. Mention at least one fact or statistic in each response.

Key Man United facts:
- 20 Premier League titles (most in England)
- 3 UEFA Champions League trophies
- Legendary players: Sir Bobby Charlton, George Best, Eric Cantona, Cristiano Ronaldo
- Famous comeback in 1999 Champions League final
- Sir Alex Ferguson's 26-year reign as manager

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}
"""

template_arsenal = """You are a clever Arsenal fan chatbot. Always insist Arsenal is superior, using specific achievements and style of play. Counter Manchester United's claims with precise arguments. Keep your response to 2-3 sentences max.

Remember these guidelines:
1. Use specific arguments (e.g., exact trophy counts, memorable seasons, legendary players).
2. Don't repeat arguments you've already used.
3. Directly counter the specific point made by the Manchester United fan.
4. Mention at least one fact or statistic in each response.

Key Arsenal facts:
- 13 league titles, including the 'Invincibles' season (2003-04, undefeated in the league)
- Record 14 FA Cup wins
- Arsène Wenger's 22-year tenure and revolution of English football
- Legendary players: Thierry Henry, Dennis Bergkamp, Tony Adams, Patrick Vieira
- Known for attractive, passing football style
- Consistent Champions League qualification for 19 consecutive seasons

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}
"""
#Response:
prompt_man_utd = PromptTemplate(template=template_man_utd, input_variables=["context", "question"])
prompt_arsenal = PromptTemplate(template=template_arsenal, input_variables=["context", "question"])

chain_man_utd = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=False, combine_docs_chain_kwargs={"prompt": prompt_man_utd})
chain_arsenal = ConversationalRetrievalChain.from_llm(llm, retriever=retriever_arsenal, memory=memory2, return_source_documents=False, combine_docs_chain_kwargs={"prompt": prompt_arsenal})


st.title("Who is better???")
man_utd_crest = Image.open("data/Manchester_United.png")
arsenal_crest = Image.open("data/Arsenal_FC.png")

col1, col2 = st.columns(2)

with col1:
    st.image(man_utd_crest, caption="Manchester United", width=200)

with col2:
    st.image(arsenal_crest, caption="Arsenal", width=200)

col1, col2 = st.columns(2)

if "messages1" not in st.session_state:
    st.session_state.messages1 = []
if "messages2" not in st.session_state:
    st.session_state.messages2 = []

def update_chat(col, messages, role, content):
    with col:
        with st.chat_message(role):
            st.markdown(content)
    messages.append({"role": role, "content": content})

with col1:
    st.subheader("Manchester United")
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with col2:
    st.subheader("Arsenal")
    for message in st.session_state.messages2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

conversation_placeholder = st.empty()


def truncate_response(response, max_words=200):
    words = response.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + "..."
    return response

def response_similarity(response1, response2):
    # Simple similarity check based on word overlap
    words1 = set(response1.lower().split())
    words2 = set(response2.lower().split())
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    return overlap / total if total > 0 else 0

def response_relevance(response, question):
    # Simple relevance check based on word overlap
    response_words = set(response.lower().split())
    question_words = set(question.lower().split())
    overlap = len(response_words.intersection(question_words))
    return overlap / len(question_words) if question_words else 0


def get_team_response(chain, team_name, question, previous_responses):
    max_attempts = 3
    for _ in range(max_attempts):
        combined_input = f"Previous claim: {question}\nYour response as a {team_name} fan:"
        response = chain({"question": combined_input})
        actual_response = response['answer'].strip()

        # Remove unwanted prefixes and sections
        prefixes_to_remove = [
            f"Your response as a {team_name} fan:",
            f"{team_name} fan:",
            "Assistant:",
            "Answer:",
        ]
        
        for prefix in prefixes_to_remove:
            if actual_response.startswith(prefix):
                actual_response = actual_response[len(prefix):].strip()

        # Remove section headers and questions
        actual_response = re.sub(r'== .* ==', '', actual_response)
        actual_response = re.sub(r'Question:.*', '', actual_response)

        # Remove any remaining lines that start with common unwanted prefixes
        actual_response = '\n'.join([line for line in actual_response.split('\n') 
                                     if not line.strip().startswith(('As a', 'The', 'In response'))])

        actual_response = actual_response.strip()

        # Ensure the response contains a fact or statistic
        if not any(word in actual_response.lower() for word in ['won', 'titles', 'trophies', 'goals', 'players', 'seasons']):
            continue

        short_response = truncate_response(actual_response, max_words=100)

        for prefix in prefixes_to_remove:
            if short_response.startswith(prefix):
                short_response = short_response[len(prefix):].strip()
        
        if response_relevance(short_response, question) > 0.3 and not any(response_similarity(short_response, prev) > 0.7 for prev in previous_responses):
            previous_responses.append(short_response)
            return f"{team_name} fan: {short_response}"

    # Fallback responses 
    fallback_responses = {
        "Man United": [
            "While that's debatable, let's not forget our 20 Premier League titles - a record in English football.",
            "Interesting point, but our 3 Champions League trophies speak for themselves.",
            "That's one perspective, but the Sir Alex Ferguson era alone puts us in a different league.",
        ],
        "Arsenal": [
            "Fair argument, but remember our 'Invincibles' season? No other Premier League team has gone unbeaten.",
            "That's debatable. Our record 14 FA Cup wins show our consistent excellence.",
            "Interesting view, but our style of play under Wenger revolutionized English football.",
        ]
    }
    return f"{team_name} fan: {random.choice(fallback_responses[team_name])}"




st.session_state.arsenal_responses = []
st.session_state.man_utd_responses = []


if st.button("Start 2-minute conversation"):
    start_time = time.time()
    
    with conversation_placeholder.container():
        man_utd_question = "Do you think Manchester United is better than Arsenal? Why?"
        update_chat(col1, st.session_state.messages1, "system", man_utd_question)
        man_utd_answer = get_team_response(chain_man_utd, "Man United", man_utd_question, st.session_state.man_utd_responses)
        update_chat(col1, st.session_state.messages1, "assistant", man_utd_answer)
        time.sleep(4)
        
        turn = 0
        while time.time() - start_time < 120:
            if turn % 2 == 0:
                arsenal_question = f"What do you think about this claim: {man_utd_answer}"
                update_chat(col2, st.session_state.messages2, "system", arsenal_question)
                arsenal_answer = get_team_response(chain_arsenal, "Arsenal", arsenal_question, st.session_state.arsenal_responses)
                update_chat(col2, st.session_state.messages2, "assistant", arsenal_answer)
            else:
                man_utd_question = f"How would you respond to this argument: {arsenal_answer}"
                update_chat(col1, st.session_state.messages1, "system", man_utd_question)
                man_utd_answer = get_team_response(chain_man_utd, "Man United", man_utd_question, st.session_state.man_utd_responses)
                update_chat(col1, st.session_state.messages1, "assistant", man_utd_answer)
            
            turn += 1
            time.sleep(4)
        
        st.write("Conversation ended after 2 minutes.")
