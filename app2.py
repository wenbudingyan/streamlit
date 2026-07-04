# import packages
import pandas as pd
from dotenv import load_dotenv
import openai
import os

import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt

# load environment variables from .env file
load_dotenv()

st.write('hello')

#from zai import ZhipuAiClient

# # 初始化客户端
# client = ZhipuAiClient(api_key="YOUR_API_KEY")

# Initialize DeepSeek client
client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"  # 推荐带/v1 兼容SDK
)

print(client)

# @st.cache_data
# def get_ai_response(user_input, temperature):
#     response = client.chat.completions.create(
#         model="deepseek-v4-pro",
#         messages=[
#             {"role": "user", "content": user_input}
#         ],
#         temperature=temperature,
#         max_tokens=300
#     )
#     return response

# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return csv_path

#add a slider for temperature
# temperature = st.slider(
#     "Model temperature",
#     min_value=0.0,
#     max_value=1.0,
#     step=0.01,
#     value=0.7,
#     help="Control temperature of the model: 0 = deterministic, 1 = very creative"
# )


# Function to get sentiment using GenAI
@st.cache_data
def get_sentiment(text, _client):
    if not text or pd.isna(text):
        return "Neutral"
    try:
        response = _client.chat.completions.create(
            model="deepseek-v4-flash",  # Use the latest chat model
            messages=[
                {"role": "system", "content":
                    "Classify the sentiment of the following review as exactly one word: Positive, Negative, or Neutral."
                     "Do not include any punctuation or additional text."
                     "Only return one single word with no extra content"},
                {"role": "user", "content": f"What's the sentiment of this review? {text}"}
            ],
            temperature=0,  # Deterministic output
            max_tokens=100  # Limit response length
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"API error: {e}")
        return "Neutral"

st.title("🔍 GenAI Sentiment Analysis Dashboard")
st.write("This is your GenAI-powered data processing app.")

# user_input = st.text_input("Enter your prompt:", "Explain generative AI in one sentence.")

import string
def clean_text(text: str) -> str:
    """
    Clean text by:
    - Removing punctuation
    - Converting to lowercase
    - Stripping leading/trailing whitespace (and collapsing multiple spaces)

    Args:
        text (str): Input string to clean.

    Returns:
        str: Cleaned string.
    """
    # Lowercase the text
    text = text.lower()

    # Remove punctuation using str.translate
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Strip leading/trailing whitespace and replace multiple spaces with single space
    text = text.strip()
    text = ' '.join(text.split())

    return text

#layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button('📥Ingest dataset'):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path)
            st.session_state["df"] = df
            st.success("dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button('🔍Parse reviews'):
        st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
        st.success("reviews cleaned")
        if "df" in st.session_state:
            try:
                with st.spinner("Analyzing sentiment..."):
                    st.session_state["df"]["Sentiment"] = st.session_state["df"]["CLEANED_SUMMARY"].apply(get_sentiment, _client=client)
                    st.success(f"Sentiment analysis completed! {st.session_state["df"]["Sentiment"].value_counts()}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please ingest the dataset first.")

# with st.spinner("AI is working..."):
#     response = get_ai_response(user_input, temperature)
#     st.write(response.choices[0].message.content)


# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)

    st.subheader("Sentiment Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(filtered_df["SENTIMENT_SCORE"],
            bins=10, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sentiment Scores')
    st.pyplot(fig)

    st.subheader("Sentiment Score Distribution")
    # Interactive Visuals with Plotly
    fig = px.histogram(
        filtered_df,
        x="SENTIMENT_SCORE",
        nbins=10,
        title="Distribution of Sentiment Scores",
        labels={"SENTIMENT_SCORE": "Sentiment Score",
                "count": "Frequency"}
    )

    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)