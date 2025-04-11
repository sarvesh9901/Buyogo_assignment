# 🏨 LLM-Powered Hotel Booking Analytics & Question Answering System

This project is developed for the Solvei8 AI/ML Internship Assignment. It provides a complete pipeline to:
- Perform analytics on hotel booking data
- Answer questions using Retrieval-Augmented Generation (RAG)
- Serve insights through a FastAPI-based REST API

---

## 📁 Project Structure

```
├── hotel_bookings.csv                # Dataset
├── main.py                           # FastAPI + RAG system
├── .env                              # API Keys
└── README.md                         # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/llm-booking-analytics.git
cd llm-booking-analytics
```

### 2. Install Dependencies
Create a virtual environment and install:
```bash
pip install -r requirements.txt
```

> `requirements.txt` content:
```text
pandas
numpy
matplotlib
seaborn
fastapi
uvicorn
sentence-transformers
faiss-cpu
python-dotenv
google-generativeai
```

### 3. Set Environment Variable
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
```

---

## 🚀 Run the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

---

## 🔌 API Endpoints

| Method | Endpoint        | Description                                |
|--------|------------------|--------------------------------------------|
| POST   | `/analytics`     | Returns hotel booking analytics            |
| POST   | `/ask`           | Asks natural language questions            |


---

## 📊 Analytics Provided

- 📈 Revenue trends over time
- ❌ Cancellation rate
- 🌍 Top countries by booking
- ⏳ Lead time distribution
- 🏨 Average price by hotel type

---

## 🧠 RAG (Retrieval-Augmented Generation)

- **Embeddings**: MiniLM via `sentence-transformers`
- **Vector Store**: FAISS
- **LLM**: Gemini (`gemini-1.5-flash`)
- **Pipeline**: Top 5 most relevant booking rows passed to LLM for answering questions

---

## ✅ Deliverables

- Cleaned and preprocessed data
- Exploratory analytics
- Vector database with embeddings
- Q&A powered by Gemini
- Fully functional FastAPI service
- some sample queries and results

---
## Sample Quesries and results
1.show me revenue for 2017
Based on the provided data, the total revenue for 2017 from canceled bookings is 300.0 (75.0 * 4).  Note that this only reflects revenue *lost* due to cancellations, not actual revenue generated in 2017.  There is no information on bookings that were not canceled in 2017.

2.Which locations had the highest booking cancellations?
Based solely on the provided data, the Resort Hotel had the highest number of booking cancellations.  The data shows multiple cancellations for the same booking ID (259) in 2015.


## 📬 Contact

For inquiries, reach out to:

**Sarvesh Karanjkar**  
📧 sarveshkaranjkar516@gmail.com
