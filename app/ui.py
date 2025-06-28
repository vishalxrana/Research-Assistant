import streamlit as st
import requests
import json

BACKEND_URL = "http://127.0.0.1:8000/api"
st.set_page_config(page_title="GenAI Research Assistant", layout="wide")


def similarity_search(query, k=10, min_score=0.25):
    """Calls the /api/similarity_search endpoint."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/similarity_search",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": query, "k": k, "min_score": min_score})
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling similarity search API: {e}")
        return None


def chat_with_llm(query, k=5, min_score=0.5):
    """Calls the /api/chat endpoint."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": query, "k": k, "min_score": min_score})
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling chat API: {e}")
        return None


st.title("GenAI Research Assistant")
st.markdown("Ask questions to our private archive of scientific journal documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the journal articles..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            response_data = chat_with_llm(prompt)
            if response_data and "answer" in response_data:
                full_response = response_data["answer"]
                
                # Display citations if available
                if "citations" in response_data and response_data["citations"]:
                    citations = ", ".join(response_data["citations"])
                    full_response += f"\n\n**Citations:**\n- {citations}"

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                error_message = "Sorry, I couldn't get a response. Please check if the backend is running."
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

