# AI-Powered Booking Analysis Dashboard

## Overview

The AI-Powered Booking Analysis Dashboard is a comprehensive tool for cleaning, analyzing, and visualizing booking data. The dashboard integrates AI-driven data analysis, interactive filtering, and insightful visualizations to explore booking trends, patterns, and statistics. It helps users interact with booking data, answer specific queries, and make data-driven decisions.

## Features

- **Data Loading & Cleaning**: Load booking data from Excel, handle missing values intelligently, and clean the data for further analysis.
- **AI-Powered Chatbot**: A generative AI chatbot using Google's Gemini API integrated into the Langchain framework, providing expert-level insights and analysis of the booking data.
- **Interactive Data Filtering**: Filter booking data by attributes such as service type, booking status, and date range for customized data exploration.
- **Visualization**: Interactive charts and graphs to visualize booking trends, distribution, revenue, and more.
- **Streamlit UI**: A user-friendly, interactive web interface for easy data exploration and interaction.

## Tech Stack

- **Python**: The main programming language for data cleaning, analysis, and AI integration.
- **Pandas**: For data loading, manipulation, and cleaning.
- **ChromaDB**: A vector database for efficient data retrieval with embeddings generated via HuggingFace models.
- **Google Gemini API**: For AI-powered chatbot responses and data analysis.
- **Langchain**: Framework for integrating the Gemini API and handling queries.
- **Plotly**: For creating interactive visualizations of the booking data.
- **Streamlit**: To build the user-friendly web interface for the dashboard.

## Installation

### Requirements

Make sure you have Python 3.8+ installed. You will also need to install the following dependencies:

```bash
pip install pandas plotly streamlit chromadb huggingface_hub google-generative-ai langchain
```

### Clone the repository

```bash
git clone https://github.com/yourusername/ai-booking-dashboard.git
cd ai-booking-dashboard
```

### Set up ChromaDB and HuggingFace models

You will need to set up ChromaDB for data storage and HuggingFace embeddings for fast and accurate data retrieval. Follow the [ChromaDB setup guide](https://www.trychroma.com) and [HuggingFace documentation](https://huggingface.co/docs).

## Usage

1. **Data Loading & Cleaning**:
   - Ensure your dataset is in Excel format and contains columns like 'Booking Date', 'Class Type', 'Instructor', 'Time Slot', 'Duration', 'Facility', and 'Theme'.
   - Run the `clean_data()` function to clean and preprocess the data.
   
2. **Running the Dashboard**:
   - Launch the Streamlit web app by running:
   ```bash
   streamlit run app.py
   ```
   - You can now interact with the dashboard through the user-friendly interface in your web browser.

3. **AI-Powered Chatbot**:
   - Ask queries like "What are the most popular booking types this month?" to get insights from the AI-powered chatbot.
   - The chatbot uses the Gemini API and Langchain to provide expert-level answers based on the cleaned booking data.

4. **Exploring Data**:
   - Use the sidebar to filter booking data by service type, booking status, and date range.
   - View booking status trends, booking type distributions, revenue analysis, and more through interactive visualizations.

## Files

- `data_cleaning.py`: Python script for cleaning and preprocessing the dataset.
- `app.py`: The main Streamlit app for the Booking Analysis Dashboard.
- `requirements.txt`: List of required dependencies for the project.
- `README.md`: Documentation file you're reading right now.

## Contributing

We welcome contributions to improve the project! Feel free to submit a pull request or open an issue to suggest improvements, report bugs, or ask questions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Pandas**: For data manipulation and cleaning.
- **Plotly**: For creating interactive visualizations.
- **Streamlit**: For building the web interface.
- **Google Gemini API**: For generative AI-powered data analysis.