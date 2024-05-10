import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as lpn
from pinecone import Pinecone as pn

# Set background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FAFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize LangChain Chatbot and OpenAI Embeddings
OPENAI_API_KEY = "TA CLEF OPENAI"

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.5)
embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Pinecone setup
pc = pn(api_key="TA CLEF PINECONE")
index = pc.Index("TON INDEX PINECONE")

# Setup for vectorstore
vectorstore = lpn(index, embed, "text")

# Initialize RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

context = "Vous êtes un chatbot spécialisé dans la préparation à l'examen IELTS. Votre rôle est de corriger les réponses écrites des étudiants, d'aider à la pratique de l'expression orale, et de fournir des traductions et des clarifications pour des phrases ou des mots difficiles en anglais. Vous avez une connaissance approfondie des critères de notation de l'IELTS et des techniques efficaces pour améliorer les compétences en anglais des étudiants. Votre tâche est d'offrir des feedbacks constructifs, de proposer des exercices ciblés et de répondre à toutes les questions que les étudiants pourraient avoir sur l'examen."

# Put Orpheus' Logo at the top of the form
st.image('/Users/othmanbensouda/PycharmProjects/Orpheus/files/orpheus.jpg')


def main():
    st.subheader(
        "Bonjour ! Je m'appelle Mailfou, votre assistant personnel dédié. Laissez-moi vous accompagner dans la préparation de votre IELTS")

    # Initialize chat history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    # Show only the last 2 or 3 messages
    message_history_limit = 3  # Set this to 2 if you want to remember only the last 2 messages
    messages_to_show = st.session_state.messages[-message_history_limit:]
    for message in messages_to_show:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Donnez moi vos directives", key="chatbot_input")

    if prompt:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Veuillez patienter quelques secondes"):
            try:
                # Prepare context by clearly formatting the previous Q&A and indicating the new question
                formatted_context = context
                for msg in st.session_state.messages[-message_history_limit:]:
                    formatted_context += f"\n\nQ: {msg['content']}" if msg[
                                                                           'role'] == 'user' else f"\n\nA: {msg['content']}"

                # Add the new question
                formatted_context += f"\n\nQ: {prompt}"

                vectorstore.similarity_search(prompt, k=3)
                response = qa.run(formatted_context)
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

                # After updating, slice the messages to keep only the last few
                st.session_state.messages = st.session_state.messages[-message_history_limit:]

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
if __name__ == "__main__":
    main()
