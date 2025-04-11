# STEP 1: DATA COLLECTION & PREPROCESSING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()
# Load data from local file
df = pd.read_csv("hotel_bookings.csv")

# Clean data
# Convert reservation_status_date to datetime
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# Fill missing values based on dataset summary
df['children'].fillna(0, inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['agent'].fillna(-1, inplace=True)
df['company'].fillna(-1, inplace=True)

# Add revenue and total_nights columns
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['revenue'] = df['adr'] * df['total_nights']

df.drop(['agent', 'company'], axis=1 , inplace=True) #droped unnessary columns as they are creating recuression error.
# STEP 2: ANALYTICS

def get_analytics():
    analytics = {}

    # Revenue trend by month
    df['month_year'] = df['reservation_status_date'].dt.to_period('M')
    revenue_by_month = df.groupby('month_year')['revenue'].sum().sort_index()
    analytics['revenue_trend'] = revenue_by_month.astype(float).round(2).to_dict()

    # Cancellation rate
    cancellation_rate = df['is_canceled'].mean() * 100
    analytics['cancellation_rate'] = round(cancellation_rate, 2)

    # Top countries by bookings
    country_distribution = df['country'].value_counts().head(10)
    analytics['top_countries'] = country_distribution.to_dict()

    # Lead time distribution
    analytics['lead_time_stats'] = df['lead_time'].describe().round(2).to_dict()

    # Average price by hotel
    avg_price = df.groupby('hotel')['adr'].mean().round(2).to_dict()
    analytics['average_price_by_hotel'] = avg_price

    return analytics


# STEP 3: RAG WITH FAISS + LLM

from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# Initialize Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel("gemini-1.5-flash")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert dataframe rows into strings for embedding
corpus = df.astype(str).apply(lambda row: ' | '.join(row), axis=1).tolist()
embeddings = embedding_model.encode(corpus, show_progress_bar=True)

# FAISS Index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def query_llm(question):
    question_embedding = embedding_model.encode([question])
    D, I = index.search(np.array(question_embedding), k=5)
    context = '\n'.join([corpus[i] for i in I[0]])

    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = llm.generate_content(prompt)
    return response.text.strip()


# STEP 4: API DEVELOPMENT

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/analytics")
def analytics():
    return get_analytics()

@app.post("/ask")
def ask_question(req: AskRequest):
    answer = query_llm(req.question)
    return {"answer": answer}




# Uncomment to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# STEP 5: EVALUATION
# Use test cases and monitor logs for response time and accuracy.
