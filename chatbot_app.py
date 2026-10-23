import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np

st.set_page_config(page_title="Stockholm Tech Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Stockholm Tech Policy Assistant")
st.caption("Ask questions about our fictional company's remote work, vacation, and hardware policies.")

KNOWLEDGE_BASE = """
Nordic Cloud Solutions AB - Employee Handbook (Stockholm Office)

1. REMOTE WORK POLICY:
Employees based in Stockholm are allowed to work remotely up to 3 days per week. 
However, Tuesdays and Thursdays are mandatory office days for team collaboration at the Kista office.
Employees must ensure a stable internet connection and be available on Slack during core hours (09:00 - 15:00 CET).

2. VACATION AND TIME OFF:
Under Swedish law, employees are entitled to 25 days of paid vacation per year. 
Nordic Cloud Solutions offers an additional 5 days, totaling 30 days. 
Vacation days must be requested at least 2 weeks in advance via the HR portal.

3. HARDWARE AND EQUIPMENT:
All new employees receive a MacBook Pro and a standard monitor. 
If you require additional equipment (e.g., mechanical keyboard, ergonomic chair), you can expense up to 3000 SEK per year without prior manager approval.
Expenses over 3000 SEK require written approval from your team lead.

4. PARENTAL LEAVE:
We fully support parental leave in accordance with Försäkringskassan. 
The company supplements the state allowance to ensure 90% of your base salary for the first 6 months of parental leave.
"""

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

with st.sidebar:
    st.header("About")
    st.markdown("This chatbot uses **RAG**. It chunks an HR policy, embeds it, and uses FAISS to retrieve context before answering.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@st.cache_resource(show_spinner=False)
def setup_rag(key):
    genai.configure(api_key=key)
    chunks = [c.strip() for c in KNOWLEDGE_BASE.split("\n\n") if c.strip()]
    
    # UPDATED: Using the new gemini-embedding-001 model
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=chunks,
        task_type="retrieval_document"
    )
    embeddings = np.array(response['embedding'], dtype=np.float32)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index

def get_answer(prompt, chunks, index):
    # UPDATED: Using the new gemini-embedding-001 model
    res = genai.embed_content(
        model="models/gemini-embedding-001",
        content=prompt,
        task_type="retrieval_query"
    )
    query_vector = np.array([res['embedding']], dtype=np.float32)
    
    _, indices = index.search(query_vector, 2)
    context = "\n\n".join([chunks[i] for i in indices[0]])
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    system_prompt = f"""
    You are a helpful HR assistant for Nordic Cloud Solutions. 
    Answer the user's question using ONLY the context provided below. 
    If the answer is not in the context, say "I don't have that information in the current policy."
    
    Context:
    {context}
    """
    
    response = model.generate_content(system_prompt + "\n\nQuestion: " + prompt)
    return response.text

if prompt := st.chat_input("e.g., How many remote days do I get?"):
    if not api_key:
        st.error("API Key missing. Please check the sidebar or Streamlit Secrets.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching policy documents..."):
                try:
                    chunks, index = setup_rag(api_key)
                    answer = get_answer(prompt, chunks, index)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
