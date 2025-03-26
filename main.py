import os
import pandas as pd
import chromadb
import uuid
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Booking Analysis Dashboard", layout="wide")
st.title("📊 Booking Data Dashboard")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY") or st.secrets["API_KEY"]

# Load dataset
DATA_PATH = "cleaned_file.csv"
df = pd.read_csv(DATA_PATH)

# Convert rows into text format for retrieval
def format_row(row):
    return (
        f"Booking ID: {row['Booking ID']}, Customer: {row['Customer Name']} ({row['Customer ID']}), "
        f"Service: {row['Service Name']} ({row['Service Type']}), Date: {row['Booking Date']}, "
        f"Booking Type: {row['Booking Type']}, Instructor: {row['Instructor']}, "
        f"Time Slot: {row['Time Slot']}, Duration: {row['Duration (mins)']} mins, "
        f"Facility: {row['Facility']}, Theme: {row['Theme']}, "
        f"Price: {row['Price']}, Status: {row['Status']}, Contact: {row['Customer Email']} / {row['Customer Phone']}."
    )

# Radio button for selecting AI Chat feature
ai_chat_selected = st.radio("Select Feature", ["Data Analysis", "AI-Powered Data Search"], horizontal=True)

# Initialize variables only when AI chat is selected
if ai_chat_selected == "AI-Powered Data Search":
    # Initialize ChromaDB only if AI search is selected
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection("booking_data")

    # Generate embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = df.apply(format_row, axis=1).tolist()
    embeddings = embedding_model.embed_documents(documents)

    # Store embeddings
    for i, doc in enumerate(documents):
        collection.add(
            ids=[str(df.loc[i, "Booking ID"])],
            embeddings=[embeddings[i]],
            metadatas=[{
                "text": doc,
                "price": df.loc[i, "Price"],
                "status": df.loc[i, "Status"],
                "service": df.loc[i, "Service Name"],
                "theme": df.loc[i, "Theme"],
                "date": df.loc[i, "Booking Date"],
                "customer": df.loc[i, "Customer Name"],
                "instructor": df.loc[i, "Instructor"],
                "facility": df.loc[i, "Facility"]
            }]
        )

# Set up Gemini API
chat_model = ChatGoogleGenerativeAI(api_key=API_KEY, model="gemini-1.5-pro", temperature=0.7)

def chat_prompt_template():
    return ChatPromptTemplate(
        messages=[
            ("system", 
             "You are a data analyst with vast experience in analyzing complex datasets. "
             "When analyzing and summarizing the most relevant booking records retrieved from the database, "
             "you should provide a detailed and insightful explanation, as an expert would. "
             "Your analysis should include patterns, trends, outliers, and any notable correlations found in the data. "
             "You must also provide context and explanations for your analysis, drawing from your broad knowledge of data analytics. "
             "Ensure responses are clear, structured, and deeply insightful, helping the user understand the significance of the data."), 

            ("human", "User Query: {human_input}"),

            ("system", 
             "Relevant Booking Data (Retrieved from Vector Database):\n"
             "{retrieved_data}\n\n"
             "Analyze the retrieved records in-depth, highlighting key trends, patterns, and outliers. "
             "Provide a detailed explanation and interpretation of the data, explaining any significant insights or findings. "
             "If the data is insufficient, indicate that explicitly and suggest potential areas to explore further or additional details that would improve the analysis.")
        ]
    )

output_parser = StrOutputParser()
chat_chain = chat_prompt_template() | chat_model | output_parser

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

session_id = get_session_id()

def chat_bot(user_input):
    retrieved_data = retrieve_relevant_data(user_input)
    retrieved_text = "\n".join(retrieved_data) if retrieved_data else "No relevant booking records found."
    response = chat_chain.invoke({"human_input": user_input, "retrieved_data": retrieved_text})
    return response, retrieved_data

# Retrieve relevant records
def retrieve_relevant_data(query, top_k=1000):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return [res["text"] for res in results["metadatas"][0]]

# --- Data Selection --- 
if ai_chat_selected == "Data Analysis":
    

    # --- Radio Button for Data Selection ---
    view_option = st.radio("Choose Data View:", ["View All Data", "Apply Filters", "View Visuals"], horizontal=True)

    # Sidebar filters (only applied if 'Apply Filters' is selected)
    st.sidebar.header("Filters")

    if "filters_applied" not in st.session_state:
        st.session_state["filters_applied"] = False
        st.session_state["service_filter"] = []
        st.session_state["status_filter"] = []
        st.session_state["theme_filter"] = []
        st.session_state["date_filter"] = []

    # Temporary filter selections (do not apply immediately)
    temp_service_filter = st.sidebar.multiselect("Select Service Type", df["Service Name"].unique(), default=st.session_state["service_filter"])
    temp_status_filter = st.sidebar.multiselect("Booking Status", df["Status"].unique(), default=st.session_state["status_filter"])
    temp_theme_filter = st.sidebar.multiselect("Select Theme", df["Theme"].unique(), default=st.session_state["theme_filter"])
    temp_date_filter = st.sidebar.date_input("Select Date Range", value=st.session_state["date_filter"])


    if st.sidebar.button("Submit"):
        st.session_state["filters_applied"] = True
        st.session_state["service_filter"] = temp_service_filter
        st.session_state["status_filter"] = temp_status_filter
        st.session_state["theme_filter"] = temp_theme_filter
        st.session_state["date_filter"] = temp_date_filter

    # --- Data Filtering ---
    if view_option == "View All Data":
        filtered_df = df.copy()
        st.write("### Booking Data")
        st.dataframe(filtered_df)

    elif view_option == "Apply Filters":
        filtered_df = df.copy()
        if st.session_state["filters_applied"]:
            if st.session_state["service_filter"]:
                filtered_df = filtered_df[filtered_df["Service Name"].isin(st.session_state["service_filter"])]
            if st.session_state["status_filter"]:
                filtered_df = filtered_df[filtered_df["Status"].isin(st.session_state["status_filter"])]
            if st.session_state["theme_filter"]:
                filtered_df = filtered_df[filtered_df["Theme"].isin(st.session_state["theme_filter"])]
            if st.session_state["date_filter"]:
                filtered_df = filtered_df[
                    (df["Booking Date"] >= str(st.session_state["date_filter"][0])) & 
                    (df["Booking Date"] <= str(st.session_state["date_filter"][-1]))
                ]
        st.write("### Filtered Booking Data")
        st.dataframe(filtered_df)

        # If there is any filtered data, generate the charts
        if not filtered_df.empty:
            # Booking Trends by Service Type
            fig0 = px.bar(
                filtered_df,
                x="Booking Date",
                y="Price",
                color="Service Name",
                title="Booking Trends by Service Type",
                labels={"Booking Date": "Booking Date", "Price": "Price"},
            )
            st.plotly_chart(fig0)

            # Distribution of Booking Types
            fig2 = px.bar(
                x=filtered_df['Booking Type'].value_counts().index,
                y=filtered_df['Booking Type'].value_counts().values,
                labels={'x': 'Booking Type', 'y': 'Count'},
                title="Distribution of Booking Types"
            )
            st.plotly_chart(fig2)

            # Price Distribution by Service Type
            price_by_service_type = filtered_df.groupby('Service Name')['Price'].sum().reset_index()
            fig3 = px.pie(price_by_service_type, names='Service Name', values='Price', 
                        title='Price Distribution by Service Type', hole=0.3)
            st.plotly_chart(fig3)

            # Revenue by Theme
            theme_price = filtered_df.groupby('Theme')['Price'].sum().reset_index()
            fig4 = px.pie(theme_price, names='Theme', values='Price', 
                        title='Revenue by Theme', hole=0.3)
            st.plotly_chart(fig4)
        else:
            st.write("No data available for the selected filters.")

    elif view_option == "View Visuals":
        st.write("### Booking Visualizations")

        # Booking Status Trends Over Time
        df.groupby('Year-Month').size().plot(marker='o', linestyle='-')
        df['Booking Date'] = pd.to_datetime(df['Booking Date'])
        df['Booking Month'] = df['Booking Date'].dt.to_period('M').astype(str)
        booking_status_counts = df.groupby(['Booking Month', 'Status'])['Booking ID'].count().unstack().reset_index()
        booking_status_melted = booking_status_counts.melt(id_vars='Booking Month', var_name='Status', value_name='Count')

        fig1 = px.line(booking_status_melted, x='Booking Month', y='Count', color='Status',
                    title='Booking Status Trends Over Time',
                    labels={"Booking Month": "Booking Month", "Count": "Number of Bookings"},
                    markers=True)

        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1)
        
        # Distribution of Booking Types
        fig2 = px.bar(
        x=df['Booking Type'].value_counts().index,
        y=df['Booking Type'].value_counts().values,
        labels={'x': 'Booking Type', 'y': 'Count'},
        title="Distribution of Booking Types"
        )
        st.plotly_chart(fig2)
        # Price Distribution by Service Type
        price_by_service_type = df.groupby('Service Name')['Price'].sum().reset_index()
        fig3 = px.pie(price_by_service_type, names='Service Name', values='Price', 
                    title='Price Distribution by Service Type', hole=0.3)
        st.plotly_chart(fig3)
        # Revenue by Theme
        theme_price = df.groupby('Theme')['Price'].sum().reset_index()
        fig4 = px.pie(theme_price, names='Theme', values='Price', 
                    title='Price Distribution by Theme', hole=0.3)
        st.plotly_chart(fig4)

# --- AI Chat ---
if ai_chat_selected == "AI-Powered Data Search":
    st.title("🔍 AI-Powered Data Search")

    # User input for the query
    user_input = st.text_input("Ask a question about dataset:")
    if user_input:
        response, retrieved_data = chat_bot(user_input)
        st.write(response)
