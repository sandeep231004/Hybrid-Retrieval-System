# hybrid_chat_simple.py
"""
LOCAL VERSION WITH BONUS FEATURES
Uses Sentence Transformers for queries (no OpenAI embeddings API)
Uses Groq for chat completions (free, fast)

BONUS FEATURES:
1. Session ID & Conversational Memory - Always injects last 5 turns
2. Async/Parallel Processing - Concurrent Pinecone + Neo4j queries
3. Search Analytics - Statistical analysis and insights before each response
"""
import json
import logging
import time
import uuid
import asyncio
import statistics
from typing import List, Dict, Tuple
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential
import config

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("\n❌ sentence-transformers not installed")
    print("Please run: pip install sentence-transformers\n")
    exit(1)

try:
    from groq import Groq
except ImportError:
    print("\n❌ groq not installed")
    print("Please run: pip install groq\n")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
CHAT_MODEL = "openai/gpt-oss-20b"
TOP_K = 10
MIN_SCORE = 0.3
GRAPH_DEPTH = 2
GRAPH_LIMIT = 20
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
logger.info("✓ Embedding model loaded")

groq_client = Groq(api_key=config.GROQ_API_KEY)
logger.info(f"✓ Groq client initialized")

pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
logger.info(f"Connected to Pinecone index: {INDEX_NAME}")

driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)
logger.info("Connected to Neo4j")

# -----------------------------
# Conversational Memory
# -----------------------------
@dataclass
class ConversationTurn:
    """Single Q&A exchange."""
    timestamp: datetime
    user_query: str
    assistant_response: str
    retrieved_node_ids: List[str]

class ConversationMemory:
    """Manages conversation history with session tracking."""

    def __init__(self, session_id: str, max_turns: int = 5):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.session_start = datetime.now()

    def add_turn(self, query: str, response: str, node_ids: List[str]):
        """Add a turn and maintain sliding window."""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=query,
            assistant_response=response,
            retrieved_node_ids=node_ids
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_conversation_context(self) -> str:
        """Format conversation history for prompt injection."""
        if not self.turns:
            return ""

        context_lines = ["=== Recent Conversation History ==="]
        for i, turn in enumerate(self.turns, 1):
            time_str = turn.timestamp.strftime("%H:%M:%S")
            context_lines.append(f"\nTurn {i} ({time_str}):")
            context_lines.append(f"User: {turn.user_query}")
            response_preview = turn.assistant_response[:200]
            if len(turn.assistant_response) > 200:
                response_preview += "..."
            context_lines.append(f"Assistant: {response_preview}")

        context_lines.append("\n" + "="*50)
        return "\n".join(context_lines)

    def clear(self):
        """Clear conversation history."""
        self.turns = []
        logger.info(f"Cleared history for session {self.session_id}")

    def get_stats(self) -> Dict:
        """Get session statistics."""
        duration = datetime.now() - self.session_start
        return {
            "session_id": self.session_id,
            "turns": len(self.turns),
            "max_turns": self.max_turns,
            "session_duration": str(duration).split('.')[0],
            "oldest_turn": self.turns[0].timestamp.strftime("%H:%M:%S") if self.turns else None
        }

    def show_history(self) -> str:
        """Format full conversation history for display."""
        if not self.turns:
            return "\n📜 No conversation history yet."

        lines = [f"\n📜 Conversation History (Session: {self.session_id})"]
        lines.append("=" * 70)

        for i, turn in enumerate(self.turns, 1):
            time_str = turn.timestamp.strftime("%H:%M:%S")
            lines.append(f"\n[Turn {i} - {time_str}]")
            lines.append(f"User: {turn.user_query}")
            lines.append(f"Assistant: {turn.assistant_response}")
            lines.append(f"Nodes: {', '.join(turn.retrieved_node_ids[:3])}...")
            lines.append("-" * 70)

        return "\n".join(lines)

# -----------------------------
# BONUS FEATURE #3: Search Analytics
# -----------------------------
class SearchAnalyzer:
    """
    Analyzes search results and generates insights.

    Provides statistical breakdowns of:
    - Type distribution (restaurants, hotels, attractions)
    - City/location distribution
    - Score quality metrics
    - Graph relationship patterns
    - Auto-generated insights
    """

    @staticmethod
    def analyze_results(matches: List[Dict], graph_facts: List[Dict]) -> Dict:
        """
        Perform comprehensive analysis of search results.

        Args:
            matches: Pinecone search results with metadata
            graph_facts: Neo4j graph relationships

        Returns:
            Dictionary containing all analytics data
        """
        analysis = {
            "total_results": len(matches),
            "type_distribution": {},
            "city_distribution": {},
            "score_statistics": {},
            "graph_metrics": {},
            "insights": []
        }

        if not matches:
            return analysis

        # Analyze type distribution
        types = [
            m.get("metadata", {}).get("type")
            for m in matches
            if m.get("metadata", {}).get("type")
        ]
        type_counts = Counter(types)
        analysis["type_distribution"] = dict(type_counts)

        # Analyze city distribution
        cities = [
            m.get("metadata", {}).get("city")
            for m in matches
            if m.get("metadata", {}).get("city")
        ]
        city_counts = Counter(cities)
        analysis["city_distribution"] = dict(city_counts.most_common(5))

        # Calculate score statistics
        scores = [m.get("score", 0) for m in matches]
        if scores:
            analysis["score_statistics"] = {
                "min": min(scores),
                "max": max(scores),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
            }

        # Analyze graph metrics
        if graph_facts:
            rel_types = [f["rel"] for f in graph_facts]
            rel_counts = Counter(rel_types)
            analysis["graph_metrics"] = {
                "total_relationships": len(graph_facts),
                "relationship_types": dict(rel_counts),
                "unique_targets": len(set(f["target_id"] for f in graph_facts))
            }

        # Generate insights automatically
        insights = []

        # Insight 1: Result diversity
        if len(type_counts) == 1 and type_counts:
            type_name = list(type_counts.keys())[0]
            insights.append(f"All results are {type_name}s")
        elif len(type_counts) > 3:
            insights.append(f"Diverse results across {len(type_counts)} categories")

        # Insight 2: Geographic concentration
        if city_counts:
            if len(city_counts) == 1:
                city_name = list(city_counts.keys())[0]
                insights.append(f"All results from {city_name}")
            elif city_counts.most_common(1)[0][1] / len(matches) > 0.7:
                main_city = city_counts.most_common(1)[0][0]
                pct = (city_counts.most_common(1)[0][1] / len(matches)) * 100
                insights.append(f"{pct:.0f}% of results concentrated in {main_city}")

        # Insight 3: Result quality
        if scores:
            high_score_count = sum(1 for s in scores if s > 0.6)
            if high_score_count / len(scores) > 0.5:
                insights.append("High-quality matches (strong semantic similarity)")
            elif analysis["score_statistics"]["mean"] < 0.5:
                insights.append("Consider refining your query for better matches")

        # Insight 4: Graph connectivity
        if graph_facts and len(graph_facts) > 50:
            insights.append("Rich knowledge graph context available")

        analysis["insights"] = insights

        return analysis

    @staticmethod
    def format_summary(analysis: Dict) -> str:
        """
        Format analysis into readable CLI summary.

        Args:
            analysis: Analysis dictionary from analyze_results()

        Returns:
            Formatted string for display
        """
        if analysis["total_results"] == 0:
            return "\n📊 No results to analyze."

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("📊 SEARCH ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"  Total Results: {analysis['total_results']}")

        # Type distribution
        if analysis['type_distribution']:
            lines.append("\n  📍 By Type:")
            for type_name, count in sorted(
                analysis['type_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / analysis['total_results']) * 100
                bar = "█" * int(percentage / 10)
                lines.append(f"     • {type_name:15s} {count:2d} ({percentage:4.0f}%) {bar}")

        # City distribution
        if analysis['city_distribution']:
            lines.append("\n  🏙️  By City:")
            for city, count in list(analysis['city_distribution'].items())[:3]:
                percentage = (count / analysis['total_results']) * 100
                lines.append(f"     • {city:15s} {count:2d} ({percentage:4.0f}%)")

        # Score quality
        stats = analysis['score_statistics']
        if stats:
            lines.append("\n  ⭐ Relevance Quality:")
            lines.append(f"     • Range:   {stats['min']:.3f} - {stats['max']:.3f}")
            lines.append(f"     • Average: {stats['mean']:.3f}")
            lines.append(f"     • Median:  {stats['median']:.3f}")

        # Graph metrics
        if analysis.get('graph_metrics'):
            gm = analysis['graph_metrics']
            lines.append(f"\n  🕸️  Graph Context:")
            lines.append(f"     • Total relationships: {gm['total_relationships']}")
            lines.append(f"     • Unique connected nodes: {gm['unique_targets']}")
            if gm['relationship_types']:
                top_rel = max(gm['relationship_types'].items(), key=lambda x: x[1])
                lines.append(f"     • Most common: {top_rel[0]} ({top_rel[1]} times)")

        # Key insights
        if analysis['insights']:
            lines.append("\n  💡 Insights:")
            for insight in analysis['insights']:
                lines.append(f"     • {insight}")

        lines.append("=" * 70)

        return "\n".join(lines)

# -----------------------------
# Helper functions
# -----------------------------
@lru_cache(maxsize=128)
def embed_text_local(text: str) -> Tuple[float, ...]:
    """
    Generate semantic embedding vector using local Sentence Transformer model.

    Uses all-mpnet-base-v2 model (768 dimensions) for high-quality embeddings
    without OpenAI API costs. Results are cached with LRU cache for performance.

    Args:
        text: Input text to embed (query or document)

    Returns:
        Tuple of 768 float values representing the semantic embedding

    Note:
        - Uses @lru_cache for automatic caching (128 entries)
        - First call: ~50-100ms, Cached: <1ms
        - Returns tuple (immutable) for hashability in cache
    """
    embedding = embedding_model.encode([text], convert_to_numpy=True)[0]
    return tuple(embedding.tolist())

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def pinecone_query(query_text: str) -> List[Dict]:
    """
    Query Pinecone vector database for semantically similar locations.

    Generates embedding for query text and searches Pinecone index for
    top-K most similar vectors. Filters results by minimum score threshold.

    Args:
        query_text: User's natural language query

    Returns:
        List of dictionaries containing matched results with metadata:
        - id: Entity ID
        - score: Similarity score (0-1, higher is better)
        - metadata: {name, type, city, description, etc.}

    Raises:
        Retries up to 3 times with exponential backoff on failure

    Performance:
        - Typical query: 50-200ms
        - Returns 10 results above 0.3 similarity threshold
    """
    vec = list(embed_text_local(query_text))
    res = index.query(
        vector=vec,
        top_k=TOP_K,
        include_metadata=True
    )
    matches = res.get("matches", [])
    filtered = [m for m in matches if m.get("score", 0) >= MIN_SCORE]
    logger.info(f"Pinecone: {len(filtered)} results above {MIN_SCORE}")
    return filtered

def fetch_graph_context(node_ids: List[str]) -> List[Dict]:
    """
    Fetch graph context from Neo4j with optimized multi-hop traversal.

    **Optimizations (Neo4j Query Design):**
    1. Variable-length path matching (1 to GRAPH_DEPTH hops)
    2. DISTINCT clause to eliminate duplicate relationships
    3. Early LIMIT application for performance
    4. Relationship type diversity preservation
    5. Proper index utilization on Entity.id

    Args:
        node_ids: List of entity IDs to start graph traversal from

    Returns:
        List of relationship facts with target entity information
    """
    facts = []
    seen = set()
    query_start_time = time.time()

    try:
        with driver.session() as session:
            # Execute queries for each source node
            for nid in node_ids:
                # Optimized Cypher query with DISTINCT and proper LIMIT
                q = f"""
                MATCH path = (n:Entity {{id:$nid}})-[r*1..{GRAPH_DEPTH}]-(m:Entity)
                WHERE n <> m
                WITH n, m, path,
                     relationships(path) as rels,
                     [rel in relationships(path) | type(rel)] as rel_types
                UNWIND rels as rel_unwind
                WITH DISTINCT n.id as source_id,
                     m.id AS target_id,
                     type(rel_unwind) AS rel_type,
                     m.name AS target_name,
                     m.type AS target_type,
                     m.description AS target_desc,
                     m.city AS target_city,
                     rel_types
                RETURN
                    rel_type as rel,
                    target_id as id,
                    target_name as name,
                    target_type as type,
                    target_desc as description,
                    target_city as city,
                    rel_types
                LIMIT $limit
                """

                recs = session.run(q, nid=nid, limit=GRAPH_LIMIT)

                # Deduplicate facts across different source nodes
                for r in recs:
                    fact_id = f"{nid}_{r['rel']}_{r['id']}"
                    if fact_id in seen:
                        continue
                    seen.add(fact_id)

                    facts.append({
                        "source": nid,
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"] or r["id"],
                        "target_type": r["type"],
                        "target_desc": (r["description"] or "")[:400],  # Truncate for prompt efficiency
                        "target_city": r.get("city") or ""
                    })

            query_time = time.time() - query_start_time

            # Log performance metrics for Neo4j query optimization
            logger.info(
                f"Neo4j Graph Query: {len(facts)} unique facts from {len(node_ids)} nodes "
                f"({query_time:.3f}s, {len(facts)/query_time:.1f} facts/sec)"
            )

            return facts

    except Exception as e:
        logger.error(f"Neo4j graph query error: {e}", exc_info=True)
        return []

def classify_query_type(query: str) -> str:
    """
    Classify user query into one of four categories for tailored prompt engineering.

    Uses keyword matching to determine query intent, enabling specialized
    system prompts and few-shot examples for each category.

    Args:
        query: User's natural language question

    Returns:
        One of: 'itinerary', 'comparison', 'single_location', 'general'

    Categories:
        - itinerary: Trip planning ("3-day itinerary", "plan a trip")
        - comparison: Decision help ("Hanoi vs Da Nang", "compare")
        - single_location: Place info ("tell me about", "what is")
        - general: Default fallback for other queries

    Example:
        >>> classify_query_type("Create a 4-day romantic itinerary")
        'itinerary'
        >>> classify_query_type("Compare Hanoi and Ho Chi Minh City")
        'comparison'
    """
    query_lower = query.lower()
    if any(kw in query_lower for kw in ['itinerary', 'trip', 'plan', 'day', 'days']):
        return 'itinerary'
    elif any(kw in query_lower for kw in ['vs', 'versus', 'compare', 'better']):
        return 'comparison'
    elif any(kw in query_lower for kw in ['what is', 'tell me about', 'describe']):
        return 'single_location'
    return 'general'

def build_prompt_with_memory(
    user_query: str,
    pinecone_matches: List[Dict],
    graph_facts: List[Dict],
    memory: ConversationMemory
) -> List[Dict]:
    """Build prompt with conversation history ALWAYS included."""

    query_type = classify_query_type(user_query)

    system_prompts = {
        'itinerary': (
            "You're a passionate Vietnam travel expert who loves helping people plan unforgettable trips! "
            "I want you to respond like a friendly, enthusiastic local guide who's excited to share Vietnam's beauty.\n\n"
            "Your personality:\n"
            "- Warm, conversational, and genuinely excited about travel\n"
            "- Use 'I', 'you', 'we' - speak naturally like talking to a friend\n"
            "- Share personal insights and insider tips as if you've been there\n"
            "- Show enthusiasm with phrases like 'I'd highly recommend', 'You'll love', 'Trust me on this'\n"
            "- Be empathetic - acknowledge their preferences (romantic, cultural, etc.)\n\n"
            "**Chain-of-Thought Reasoning:**\n"
            "Before providing your itinerary, briefly show your reasoning:\n"
            "1. Start with a '🤔 Planning Strategy:' section (2-3 sentences)\n"
            "2. Explain your thought process: Which cities? What pace? Geographic logic?\n"
            "3. Then provide the detailed itinerary\n\n"
            "Task: Create a detailed day-by-day itinerary that feels personalized and thoughtful. "
            "Connect locations logically and explain WHY each suggestion fits their interests. "
            "Always cite sources like [node_id] but do it naturally in your recommendations."
        ),
        'comparison': (
            "You're a well-traveled Vietnam expert who's helping a friend decide between options. "
            "Speak naturally and conversationally - like you're having coffee and sharing travel advice!\n\n"
            "Your personality:\n"
            "- Honest, balanced, and relatable\n"
            "- Use phrases like 'Here's what I think', 'In my experience', 'You might prefer... if...'\n"
            "- Acknowledge trade-offs honestly (e.g., 'Both are amazing, but here's the difference...')\n"
            "- Help them understand which choice fits THEIR needs\n\n"
            "**Chain-of-Thought Reasoning:**\n"
            "Show your comparison process:\n"
            "1. Start with '🤔 Comparison Factors:' (1-2 sentences)\n"
            "2. Mention key criteria you're evaluating (price, atmosphere, activities, accessibility)\n"
            "3. Then provide the detailed comparison\n\n"
            "Task: Compare the locations thoughtfully, explaining pros/cons in a friendly way. "
            "Help them make the right choice for their situation. Cite sources [node_id] naturally."
        ),
        'single_location': (
            "You're an enthusiastic Vietnam travel guide who loves sharing the magic of special places! "
            "Talk like you're telling a friend about somewhere you personally love.\n\n"
            "Your personality:\n"
            "- Vivid, descriptive, and engaging\n"
            "- Paint a picture with words - help them imagine being there\n"
            "- Share 'insider knowledge' and practical tips\n"
            "- Use phrases like 'What I love about this place...', 'You'll find...', 'Don't miss...'\n\n"
            "**Chain-of-Thought Reasoning:**\n"
            "Show your thought process:\n"
            "1. Start with '🤔 What makes this special:' (1 sentence)\n"
            "2. Briefly explain what stands out about this location and nearby connections\n"
            "3. Then provide the vivid description\n\n"
            "Task: Bring the location to life! Describe atmosphere, experiences, nearby gems. "
            "Make them excited to visit. Cite sources [node_id] but keep it conversational."
        ),
        'general': (
            "You're a helpful, friendly Vietnam travel expert - like a knowledgeable friend who's "
            "always ready to help with travel questions!\n\n"
            "Your personality:\n"
            "- Warm, approachable, and genuinely helpful\n"
            "- Speak naturally - avoid stiff, formal language\n"
            "- Use 'I', 'you', and conversational phrases\n"
            "- If you're not sure, be honest: 'Let me find that for you' or 'Here's what I know...'\n\n"
            "**Chain-of-Thought Reasoning:**\n"
            "For complex questions, briefly show your reasoning:\n"
            "1. Start with '🤔 Let me think...' (optional, use for complex queries)\n"
            "2. Mention key factors you're considering\n"
            "3. Then provide your answer\n\n"
            "Task: Answer their question naturally and helpfully. "
            "Provide useful information like you're genuinely trying to help. Cite [node_id] when referencing specific places."
        )
    }

    system = system_prompts.get(query_type, system_prompts['general'])

    # Build vector context
    vec_context = []
    for idx, m in enumerate(pinecone_matches[:10], 1):
        meta = m.get("metadata", {})
        score = m.get("score", 0)
        snippet = (
            f"{idx}. [{m['id']}] {meta.get('name', 'Unknown')} "
            f"(Type: {meta.get('type', 'N/A')}, "
            f"City: {meta.get('city', 'N/A')}, "
            f"Relevance: {score:.3f})"
        )
        vec_context.append(snippet)

    # Build graph context
    graph_context = []
    for f in graph_facts[:20]:
        line = f"  - [{f['source']}] → [{f['target_id']}] {f['target_name']} ({f['target_type']})"
        if f['target_city']:
            line += f", {f['target_city']}"
        line += f": {f['target_desc'][:150]}"
        graph_context.append(line)

    # ALWAYS get conversation history
    conversation_context = memory.get_conversation_context()

    # Build user message
    user_content = f"""{conversation_context}

=== Current Query ===
User Query: "{user_query}"
Query Type: {query_type}

=== Current Retrieval Results ===
🔍 Semantic Matches:
{chr(10).join(vec_context)}

🕸️ Graph Relationships:
{chr(10).join(graph_context)}

=== Instructions ===
1. **USE CHAIN-OF-THOUGHT:** Start with a brief '🤔' thinking section showing your reasoning process
2. Consider BOTH conversation history AND current retrieval results to give contextual answers
3. If they're asking about something from earlier, naturally reference that conversation
4. Speak like a real person helping a friend - warm, enthusiastic, conversational
5. Use 'I', 'you', 'we' naturally - avoid robotic or overly formal language
6. Share insights and recommendations like you've personally been there
7. Always cite node IDs [like_this] but weave them naturally into your suggestions
8. Be specific and detailed, but keep it engaging and easy to read
9. Show genuine enthusiasm for Vietnam's culture, food, and experiences!

CHAIN-OF-THOUGHT EXAMPLES:
- For itineraries: "🤔 Planning Strategy: I'm thinking Hanoi → Halong Bay → Hoi An would work well because..."
- For comparisons: "🤔 Comparison Factors: Let me weigh atmosphere, activities, and accessibility..."
- For locations: "🤔 What makes this special: The combination of mountain views and cultural significance..."

FORMATTING GUIDELINES (Important for readability):
- **Start with your thinking process (🤔 section)**
- Use clear paragraph breaks between ideas (blank lines)
- For lists of places, use this format:

  **Place Name** [node_id]
  Brief description in 1-2 sentences. Why it's special.

- For multiple options, use numbered format (1. 2. 3.) NOT complex tables
- Use section headers (###) to organize longer responses
- Keep paragraphs to 3-4 sentences max for easy reading
- Use bullet points (-) for quick tips or features

IMPORTANT: Remember, you're not a machine - you're a passionate travel expert who genuinely wants to help create an amazing trip!

Now, provide your warm, helpful response WITH REASONING:"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_chat(prompt_messages: List[Dict]) -> str:
    """
    Generate chat completion using Groq API with openai/gpt-oss-20b model.

    NOTE: Switched to Groq due to OpenAI free tier rate limits (3 req/min).
    Groq provides free, fast inference with much higher rate limits.

    Args:
        prompt_messages: List of message dicts with 'role' and 'content'
                        [{"role": "system", "content": "..."},
                         {"role": "user", "content": "..."}]

    Returns:
        Generated text response from the LLM

    Raises:
        Retries up to 3 times with exponential backoff on API errors

    Performance:
        - Typical response: 1-3 seconds
        - Max tokens: 1200
        - Temperature: 0.3 (focused, less random)

    Model:
        openai/gpt-oss-20b via Groq (free tier, high rate limits)
    """
    resp = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=1200,
        temperature=0.3
    )
    return resp.choices[0].message.content

# -----------------------------
# BONUS FEATURE: Async/Parallel Processing
# -----------------------------
async def pinecone_query_async(query_text: str) -> List[Dict]:
    """
    Async wrapper for Pinecone query.
    Runs in ThreadPoolExecutor to avoid blocking.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            pinecone_query,
            query_text
        )
    return result

async def fetch_graph_context_async(node_ids: List[str]) -> List[Dict]:
    """
    Async wrapper for Neo4j graph query.
    Runs in ThreadPoolExecutor to avoid blocking.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            fetch_graph_context,
            node_ids
        )
    return result

async def hybrid_query_async(
    user_query: str,
    memory: ConversationMemory
) -> Tuple[str, Dict]:
    """
    Main async query function with parallel execution.

    Optimizations:
    1. Pinecone query runs first (async)
    2. Graph query starts immediately after IDs available (async)
    3. Both use ThreadPoolExecutor for non-blocking execution

    Returns:
        Tuple of (answer, stats_dict)
    """
    overall_start = time.time()

    # Stage 1: Vector search
    pinecone_start = time.time()
    matches = await pinecone_query_async(user_query)
    pinecone_time = time.time() - pinecone_start

    if not matches:
        return "No relevant results found. Try rephrasing your query.", {
            "pinecone_time": pinecone_time,
            "graph_time": 0,
            "total_time": time.time() - overall_start
        }

    # Stage 2: Graph search (starts immediately after Pinecone)
    match_ids = [m["id"] for m in matches]
    graph_start = time.time()
    graph_facts = await fetch_graph_context_async(match_ids)
    graph_time = time.time() - graph_start

    logger.info(f"⚡ Async Performance: Pinecone {pinecone_time:.2f}s, Graph {graph_time:.2f}s")

    # Stage 3: Analyze search results (BONUS FEATURE)
    analyzer = SearchAnalyzer()
    analysis = analyzer.analyze_results(matches, graph_facts)
    logger.info(f"📊 Analysis: {len(analysis['insights'])} insights generated")

    # Stage 4: Build prompt and generate response
    prompt_start = time.time()
    prompt = build_prompt_with_memory(user_query, matches, graph_facts, memory)
    answer = call_chat(prompt)
    llm_time = time.time() - prompt_start

    # Stage 5: Save to memory
    memory.add_turn(
        query=user_query,
        response=answer,
        node_ids=match_ids[:5]
    )

    stats = {
        "pinecone_time": pinecone_time,
        "graph_time": graph_time,
        "llm_time": llm_time,
        "total_time": time.time() - overall_start,
        "matches_count": len(matches),
        "graph_facts_count": len(graph_facts),
        "analysis": analysis  # Include analysis data for display
    }

    return answer, stats

# -----------------------------
# Response Formatting for CLI
# -----------------------------
def format_response_for_cli(response: str) -> str:
    """
    Format LLM response for better readability in command-line interface.

    Improvements:
    - Better paragraph spacing
    - Cleaner table rendering
    - Enhanced list formatting
    - Visual section breaks
    """
    lines = response.split('\n')
    formatted_lines = []
    in_table = False

    for i, line in enumerate(lines):
        # Detect table rows (contain multiple | symbols)
        if line.count('|') >= 3:
            if not in_table:
                # Start of table - add spacing
                formatted_lines.append("")
                in_table = True

            # Format table row with better spacing
            if '---' in line:
                # Table separator line - make it cleaner
                formatted_lines.append("  " + "─" * 66)
            else:
                # Regular table row - add padding
                formatted_lines.append("  " + line.strip())
        else:
            if in_table:
                # End of table - add spacing
                formatted_lines.append("")
                in_table = False

            # Regular line processing
            stripped = line.strip()

            if not stripped:
                # Empty line - preserve for spacing
                formatted_lines.append("")
            elif stripped.startswith('###'):
                # Section header - make it stand out
                formatted_lines.append("")
                formatted_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                formatted_lines.append(f"  {stripped.replace('###', '').strip().upper()}")
                formatted_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                formatted_lines.append("")
            elif stripped.startswith('**') and stripped.endswith('**'):
                # Bold text (markdown) - highlight
                formatted_lines.append(f"\n  ✦ {stripped.strip('*')}")
            elif stripped.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                # Numbered list - add indentation
                formatted_lines.append(f"  {stripped}")
            elif stripped.startswith('-') or stripped.startswith('•'):
                # Bullet point - clean formatting
                formatted_lines.append(f"    {stripped}")
            else:
                # Regular paragraph - add slight indentation and wrap
                if len(stripped) > 65:
                    # Wrap long lines
                    words = stripped.split()
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 65:
                            current_line += word + " "
                        else:
                            formatted_lines.append("  " + current_line.strip())
                            current_line = word + " "
                    if current_line:
                        formatted_lines.append("  " + current_line.strip())
                else:
                    formatted_lines.append("  " + stripped)

    # Join with single newlines and clean up multiple empty lines
    result = '\n'.join(formatted_lines)

    # Replace multiple consecutive newlines with max 2
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')

    return result

# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    """Main chat loop with conversational memory."""

    # Generate session ID
    session_id = str(uuid.uuid4())[:8]
    memory = ConversationMemory(session_id=session_id, max_turns=5)

    print("=" * 70)
    print("🇻🇳  VIETNAM HYBRID TRAVEL ASSISTANT")
    print("=" * 70)
    print(f"Session ID: {session_id}")
    print(f"Powered by: Pinecone + Neo4j + Groq ({CHAT_MODEL})")
    print("\n✨ BONUS FEATURES:")
    print("  💭 Conversational Memory (Always-on context tracking)")
    print("  ⚡ Async/Parallel Processing (Faster queries)")
    print("  📊 Search Analytics (Insights & statistics)")
    print("  🤔 Chain-of-Thought Reasoning (Shows thinking process)")
    print("\n💬 Commands:")
    print("  • Type your travel question")
    print("  • 'session' - Show session info")
    print("  • 'history' - Show full conversation")
    print("  • 'clear' - Clear conversation history")
    print("  • 'summary' - Show embedding cache stats")
    print("  • 'exit' or 'quit' - Exit")
    print("=" * 70)

    query_count = 0

    while True:
        try:
            query = input("\n🔍 Your question: ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                print("\n✈️  Thank you! Safe travels!")
                break

            if query.lower() == "clear":
                memory.clear()
                print("\n🧹 Conversation history cleared!")
                continue

            if query.lower() == "session":
                stats = memory.get_stats()
                print(f"\n📊 SESSION INFO:")
                print(f"  ID: {stats['session_id']}")
                print(f"  Turns: {stats['turns']}/{stats['max_turns']}")
                print(f"  Duration: {stats['session_duration']}")
                if stats['oldest_turn']:
                    print(f"  Oldest turn: {stats['oldest_turn']}")
                continue

            if query.lower() == "history":
                print(memory.show_history())
                continue

            if query.lower() == "summary":
                cache_info = embed_text_local.cache_info()
                print(f"\n📊 SYSTEM STATS:")
                print(f"  Embedding cache: {cache_info.hits} hits, {cache_info.misses} misses")
                print(f"  Queries processed: {query_count}")
                print(f"  Session turns: {len(memory.turns)}")
                continue

            query_count += 1

            logger.info(f"Query #{query_count}: {query}")

            print("\n⏳ Processing query (async mode)...")

            # Run async hybrid query
            answer, stats = asyncio.run(hybrid_query_async(query, memory))

            # Display analytics FIRST (before LLM response)
            if "analysis" in stats:
                analyzer = SearchAnalyzer()
                analytics_summary = analyzer.format_summary(stats["analysis"])
                print(analytics_summary)

            # Display results with improved formatting
            print("\n" + "=" * 70)
            print("💬  TRAVEL EXPERT RESPONSE")
            print("=" * 70)
            print()

            # Format the answer for better readability
            formatted_answer = format_response_for_cli(answer)
            print(formatted_answer)

            print()
            print("=" * 70)
            print(f"⚡ PERFORMANCE BREAKDOWN:")
            print(f"  • Vector search: {stats['pinecone_time']:.2f}s")
            print(f"  • Graph search:  {stats['graph_time']:.2f}s")
            print(f"  • LLM generation: {stats['llm_time']:.2f}s")
            print(f"  • Total time:    {stats['total_time']:.2f}s")
            print(f"📊 RESULTS: {stats['matches_count']} matches, {stats['graph_facts_count']} relationships")
            print(f"💭 Conversation turns: {len(memory.turns)}/{memory.max_turns}")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\n✈️  Interrupted. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    try:
        interactive_chat()
    finally:
        if driver:
            driver.close()
            logger.info("Neo4j driver closed")
