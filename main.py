import os
import json
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from openai import OpenAI

# Set environment variables
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# Get API keys from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone
pc = PineconeClient(api_key=pinecone_api_key)

# Define index name and check if it exists
index_name = "travelinfo"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load the JSON data from file and upsert to Pinecone if index is empty
if index.describe_index_stats()['total_vector_count'] == 0:
    with open("C:\\Users\\hardi\\OneDrive\\Desktop\\HS\\Gen AI Final Project\\data.json", "r") as file:
        data = json.load(file)
        vectors = []
        for i, item in enumerate(data):
            text = f"location: {item['location']}. attractions: {', '.join(item['attractions'])}"
            vector = embeddings.embed_documents([text])[0]
            vectors.append({"id": str(i), "values": vector, "metadata": item})
        index.upsert(vectors=vectors, namespace="locations-attraction-namespace")

# Initialize LangChain components
vectorstore = Pinecone(index=index, embedding=embeddings, text_key='location', namespace="locations-attraction-namespace")
retriever = vectorstore.as_retriever()

# Streamlit app
st.title("TravelCompanion.ai")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
- To generate a response type in a location. eg. Berlin, Germany
                    """)
st.sidebar.header("About TravelCompanion.ai")
st.sidebar.info("""
This app leverages advanced AI to provide you with insights and recommendations based on attractions around the world. Simply enter a location to discover key attractions and receive personalized insights to enhance your travel experience.
""")
st.sidebar.markdown("""By Hardik Sodhani""")

# User input for location
st.write("Enter your location below and get attraction suggestions:")
user_input = st.text_input("Location")

if user_input:
    result = retriever.get_relevant_documents(user_input)
    found_match = False
    for doc in result:
        location = doc.metadata.get('location')
        attractions = ', '.join(doc.metadata.get('attractions', []))
        messages = [
            {"role": "system", "content": "You are an AI trained to provide tourism insights."},
            {"role": "user", "content": f"Analyze the attractions in {location}: {attractions}"}
        ]
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=500)
        if response.choices:
            insights = response.choices[0].message.content
        else:
            insights = "No insights were generated."
        st.subheader(f"Insights for {user_input}")
        st.write(insights)
        found_match = True
        break

    if not found_match:
        st.write("Sorry, no relevant information found for that location.")
