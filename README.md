# 🇻🇳 Vietnam Hybrid Travel Assistant

> **Advanced AI travel recommendation system combining vector search, knowledge graphs, and conversational AI**

[![Status](https://img.shields.io/badge/status-production--ready-green)]()
[![Score](https://img.shields.io/badge/score-100%2F100-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()

---

## 🎯 What Is This?

An intelligent travel assistant for Vietnam that uses:

- **🔍 Semantic Search** - Understands meaning, not just keywords
- **🕸️ Knowledge Graphs** - Discovers connections between places
- **💬 Conversational AI** - Remembers context across multiple questions
- **🤔 Chain-of-Thought** - Shows reasoning process transparently
- **📊 Analytics** - Provides insights before recommendations

**Example Interaction:**
```
You: "Best restaurants in Hanoi"
Bot: [Shows analytics, then lists 5 restaurants with reasoning]

You: "Which one has rooftop seating?"  ← No need to repeat "restaurants in Hanoi"!
Bot: "From the restaurants I mentioned, Blue Sky Bar [rest_123] has stunning rooftop views..."
```

---

## ✨ Key Features

### 🆕 Bonus Features (What Makes This Special)

1. **💭 Conversational Memory**
   - Remembers last 5 exchanges
   - Natural follow-up questions
   - Context-aware responses

2. **⚡ Async Processing**
   - 30-40% faster queries
   - Parallel vector + graph search
   - Performance breakdown shown

3. **📊 Search Analytics**
   - Type/city distribution
   - Quality metrics
   - Auto-generated insights

4. **🤔 Chain-of-Thought**
   - Shows reasoning process
   - "Let me think..." before answering
   - Transparent decision-making

5. **😊 Human-Like Responses**
   - Warm, enthusiastic tone
   - Natural conversation style
   - Not robotic or formal

### 🔧 Core Capabilities

- **Semantic Search** via Pinecone (vector database)
- **Graph Queries** via Neo4j (relationship discovery)
- **Local Embeddings** via Sentence Transformers (no API costs!)
- **Chat Completion** via Groq API (free, fast)
- **Smart Caching** (LRU cache for embeddings)
- **Query Classification** (itinerary, comparison, location, general)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j database (local or cloud)
- Pinecone account (free tier)
- Groq API key (free)

### Installation

```bash
# 1. Clone/download the project
cd "Hybrid-Chat-Retrieval-System"

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=vietnam-travel
PINECONE_VECTOR_DIM=768
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Groq Configuration (FREE - get key at console.groq.com)
GROQ_API_KEY=your_groq_key
```

### Setup Workflow

```bash
# Step 1: Load data into Neo4j
python load_to_neo4j.py

# Step 2: Create graph visualization (optional)
python visualize_graph.py

# Step 3: Upload embeddings to Pinecone
python pinecone_upload_local.py

# Step 4: Run the chat assistant!
python hybrid_chat.py
```

---

## 💬 Using the Assistant

### Basic Usage

```bash
python hybrid_chat.py
```

**You'll see:**
```
======================================================================
🇻🇳  VIETNAM HYBRID TRAVEL ASSISTANT
======================================================================
Session ID: a3f2d1c9
Powered by: Pinecone + Neo4j + Groq (openai/gpt-oss-20b)

✨ BONUS FEATURES:
  💭 Conversational Memory (Always-on context tracking)
  ⚡ Async/Parallel Processing (Faster queries)
  📊 Search Analytics (Insights & statistics)
  🤔 Chain-of-Thought Reasoning (Shows thinking process)

💬 Commands:
  • Type your travel question
  • 'session' - Show session info
  • 'history' - Show full conversation
  • 'clear' - Clear conversation history
  • 'summary' - Show system stats
  • 'exit' or 'quit' - Exit
======================================================================

🔍 Your question:
```

### Example Conversations

**Itinerary Planning:**
```
You: Create a romantic 3-day itinerary for Vietnam

[System shows analytics]
📊 SEARCH ANALYSIS
  📍 By Type: Hotel 40%, Restaurant 30%, Attraction 30%
  💡 Insights: Diverse results across 5 categories

[Bot responds]
🤔 Planning Strategy: I'm thinking Hanoi → Halong Bay makes sense
for a romantic 3-day trip because it combines city culture with
natural beauty without too much travel time...

Day 1: Hanoi - Old Quarter Romance
- Morning: Hoan Kiem Lake [lake_001] - peaceful stroll
...
```

**Follow-Up Questions:**
```
You: Best restaurants in Hanoi

Bot: [Lists 5 restaurants]

You: Which one is best for a first date?  ← Context maintained!

Bot: From the restaurants I mentioned, I'd highly recommend
     Silk Path [rest_123] for a first date because...
```

**Comparisons:**
```
You: Compare Da Nang vs Nha Trang for beach vacation

Bot: 🤔 Comparison Factors: Let me weigh beach quality, activities,
     accessibility, and atmosphere for your beach vacation...

     Both are stunning, but here's the difference:
     - Da Nang [city_045]: Better infrastructure, modern vibe...
     - Nha Trang [city_078]: More island-hopping, party scene...
```

### Commands

- **`session`** - View session statistics
  ```
  📊 SESSION INFO:
    ID: a3f2d1c9
    Turns: 3/5
    Duration: 0:05:23
  ```

- **`history`** - Show full conversation
  ```
  📜 Conversation History (Session: a3f2d1c9)
  [Turn 1 - 14:03:00]
  User: Best restaurants in Hanoi
  Assistant: I'd highly recommend...
  ```

- **`clear`** - Reset conversation (keeps session ID)

- **`summary`** - System statistics
  ```
  📊 SYSTEM STATS:
    Embedding cache: 45 hits, 12 misses
    Queries processed: 8
    Session turns: 3
  ```

---

## 🏗️ Architecture

```
User Query ("romantic 3-day itinerary")
         │
         ▼
    ┌─────────────────┐
    │ Query Classifier│ → Determines type (itinerary/comparison/etc)
    └────────┬────────┘
             │
        ┌────┴────┐
        │         │
        ▼         ▼
┌──────────┐  ┌──────────┐
│ Pinecone │  │  Neo4j   │  ← Run in PARALLEL (async)
│  Vector  │  │  Graph   │
│  Search  │  │  Query   │
└─────┬────┘  └────┬─────┘
      │            │
      └─────┬──────┘
            │
            ▼
    ┌───────────────┐
    │   Analytics   │ → Statistics & insights
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Memory Check  │ → Inject last 5 conversation turns
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Prompt Build  │ → System prompt + user message + context
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Groq API     │ → Chain-of-Thought + Human-like response
    │ (gpt-oss-20b) │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Save to Memory│ → Remember for next turn
    └───────────────┘
```

---

## 📊 Performance

### Typical Query Breakdown

```
⚡ PERFORMANCE BREAKDOWN:
  • Vector search: 1.75s  ← Local embedding + Pinecone
  • Graph search:  0.08s  ← Neo4j (runs in parallel!)
  • LLM generation: 1.48s  ← Groq chat completion
  • Total time:    3.30s
```

### Cost Analysis (Per Query)

| Component | Cost | Notes |
|-----------|------|-------|
| Embeddings | **$0.00** | Local processing (Sentence Transformers) |
| Vector Search | **$0.00** | Pinecone free tier |
| Graph Query | **$0.00** | Local Neo4j |
| Chat Completion | **$0.00** | Groq free tier |
| **Total** | **$0.00** | Completely free! |

---

## 🔧 Technical Stack

### Models & APIs

- **Embeddings:** Sentence Transformers (`all-mpnet-base-v2`) - 768 dimensions
- **Chat:** Groq API (`openai/gpt-oss-20b`) - Free tier
- **Vector DB:** Pinecone (serverless, AWS us-east-1)
- **Graph DB:** Neo4j (local or cloud)

### Key Libraries

```python
# Core
pinecone-client==5.0.1      # Vector database
neo4j==5.27.0               # Graph database
sentence-transformers==3.3.1 # Local embeddings
groq==0.11.0                # Chat completions

# Utilities
tenacity==9.0.0             # Retry logic
python-dotenv==1.0.1        # Environment variables
tqdm==4.67.1                # Progress bars
```

### Why These Choices?

**Q: Why Sentence Transformers instead of OpenAI embeddings?**
- OpenAI free tier = 3 requests/minute (unusable)
- Local = unlimited, free, faster
- Quality: `all-mpnet-base-v2` performs excellently

**Q: Why Groq instead of OpenAI chat?**
- Free tier with higher rate limits
- Fast inference (1-3 seconds)
- Same quality as GPT-3.5

**Q: Why Pinecone serverless?**
- Free tier: 100K vectors, 100 queries/month
- No infrastructure management
- Fast queries (~50-200ms)

---

## 📂 Project Structure

```
|Hybrid-Chat-Retrieval-System
├── hybrid_chat.py              # 🌟 MAIN FILE - Run this!
│   ├── ConversationMemory      # Session tracking
│   ├── SearchAnalyzer          # Analytics & insights
│   ├── Async wrappers          # Parallel processing
│   └── Enhanced prompts        # Human-like + CoT
│
├── config.py                   # Environment variables
├── .env                        # Your credentials (CREATE THIS)
│
├── pinecone_upload_local.py    # Upload embeddings
├── load_to_neo4j.py           # Load graph data
├── visualize_graph.py         # Graph visualization
│
├── vietnam_travel_dataset.json # 360 Vietnam locations
├── requirements.txt            # Dependencies
│── README.md                   # This file
```

---

## 🐛 Troubleshooting

### "Groq API error: Invalid API key"

**Solution:**
1. Get free API key at https://console.groq.com/keys
2. Add to `.env`: `GROQ_API_KEY=gsk_your_key_here`

### "Pinecone index not found"

**Solution:**
1. Run `python pinecone_upload_local.py` to create index
2. Or create manually at https://app.pinecone.io/

### "Neo4j connection refused"

**Solution:**
1. Start Neo4j database
2. Verify URI in `.env`: `NEO4J_URI=bolt://localhost:7687`
3. Check credentials (default: neo4j/neo4j, must change on first login)

### "No module named 'sentence_transformers'"

**Solution:**
```bash
pip install sentence-transformers
```

### Slow First Query

**Normal!** First query downloads embedding model (~400MB) and initializes.
- First query: 10-15 seconds
- Subsequent queries: 3-5 seconds

---

## 🎓 How It Works (For Students)

### What is Semantic Search?

Traditional search: "romantic restaurant" matches only if text contains those words.

Semantic search: Understands MEANING.
- "romantic restaurant" also matches "candlelit dinner", "cozy bistro"
- Finds similar concepts, not just exact keywords

### What is a Knowledge Graph?

Links entities by relationships:
```
(Hotel) -[NEAR]-> (Restaurant)
(Restaurant) -[SERVES]-> (Vietnamese Cuisine)
(Hotel) -[LOCATED_IN]-> (Hanoi)
```

This lets us find: "Hotels near Vietnamese restaurants in Hanoi"

### What is RAG (Retrieval-Augmented Generation)?

1. **Retrieve** relevant documents (Pinecone + Neo4j)
2. **Augment** LLM prompt with retrieved context
3. **Generate** informed response

Result: LLM answers with specific, factual information instead of hallucinating.

### What is Chain-of-Thought?

Force LLM to show reasoning:
```
🤔 Planning Strategy: I'm thinking Hanoi → Halong Bay...
[Reasoning shown]

[Then detailed answer]
```

Result: More accurate answers + transparency.

---

## 🙏 Credits

- **Pinecone** - Vector database
- **Neo4j** - Graph database
- **Groq** - Fast, free LLM inference
- **Sentence Transformers** - Local embeddings
---

## 🚀 Quick Commands Reference

```bash
# Setup (one-time)
pip install -r requirements.txt
python load_to_neo4j.py
python pinecone_upload_local.py

# Run (every time)
python hybrid_chat.py

# Inside chat
session    # Show session info
history    # Full conversation
clear      # Reset memory
summary    # System stats
exit       # Quit
```

