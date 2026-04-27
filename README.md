# 🌍 Agentic Trip Planner

An AI-powered travel planning application built with Streamlit and GPT-4o. The app uses an agentic loop to search real venues via OpenStreetMap, retrieve travel guides from Wikivoyage, and generate personalized multi-day itineraries with interactive maps.

---

## 📸 Screenshot

> _Add a screenshot of your running app here._  
> Suggested: `docs/screenshot.png`

---

## ✨ Key Features

- **AI Agent Orchestration** — GPT-4o autonomously calls tools to search POIs and retrieve guides before writing the itinerary
- **Real Venue Search** — Queries OpenStreetMap's Overpass API for restaurants, parks, museums, and more by name, category, or cuisine type
- **Interactive Map** — PyDeck-powered dark map with day-by-day filtering and hover tooltips
- **Wikivoyage RAG** — TF-IDF retrieval over Wikivoyage articles grounds the itinerary in real local knowledge
- **Itinerary Refinement** — Modify specific days or the entire trip using natural language
- **Feedback System** — Upvote/downvote locations to influence future recommendations via a boost score
- **Fast Mode** — Limits agent steps for quicker previews
- **Configurable Sidebar** — Model selection (GPT-4o / GPT-4o-mini), max steps, and fast mode toggle
- **Agent Trace Viewer** — Shows each tool call and timing for full transparency

---

## 🏗️ Architecture

```
User Input (City, Days, Interests, Constraints)
        │
        ▼
┌───────────────────┐
│   Streamlit UI    │  ← Session state, sidebar config
└────────┬──────────┘
         │ prompt
         ▼
┌───────────────────┐
│  GPT-4o Agent     │  ← run_agent_loop()
│  (Agentic Loop)   │
└──┬─────────────┬──┘
   │ tool calls  │ tool calls
   ▼             ▼
┌──────────┐  ┌─────────────────┐
│ Overpass │  │  Wikivoyage API │
│ API      │  │  + TF-IDF RAG   │
│ (OSM)    │  │                 │
└──────────┘  └─────────────────┘
   │ POI data    │ travel context
   └──────┬──────┘
          ▼
┌───────────────────┐
│  JSON Itinerary   │  ← itinerary_text + locations[]
└────────┬──────────┘
         ▼
┌───────────────────┐     ┌──────────────────┐
│  PyDeck Map       │     │  Feedback System │
│  (ScatterplotLayer│     │  poi_feedback.   │
│   + day filter)   │     │  jsonl           │
└───────────────────┘     └──────────────────┘
```

**Data flow summary:**
1. User submits trip specs → validated via Nominatim geocoding
2. Agent loop calls `search_pois` (Overpass) and `retrieve_guides` (Wikivoyage) autonomously
3. GPT-4o synthesizes results into a structured JSON with `itinerary_text` and `locations`
4. Coordinates are resolved through a 3-tier fallback: poi_cache → map_points → city center
5. Map renders via PyDeck; feedback scores are persisted to a local JSONL file

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/agentic-trip-planner.git
cd agentic-trip-planner

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### requirements.txt

```
streamlit>=1.32.0
openai>=1.12.0
requests>=2.31.0
pandas>=2.0.0
pydeck>=0.8.0
scikit-learn>=1.3.0
```

### Environment Variables (for deployment)

Create a `.env` file or set these in your deployment platform:

```
OPENAI_API_KEY=sk-...
```

> ⚠️ Never hardcode API keys. The app accepts the key via the sidebar input at runtime.

---

## 🔑 API Requirements & Costs

| API | Purpose | Cost | Rate Limit |
|-----|---------|------|-----------|
| OpenAI GPT-4o | Itinerary generation | ~$0.01–0.05 per itinerary | 500 RPM (Tier 1) |
| OpenAI GPT-4o-mini | Fast mode generation | ~$0.001–0.005 per itinerary | 500 RPM (Tier 1) |
| Nominatim (OSM) | Geocoding city names | Free | 1 req/sec |
| Overpass API (OSM) | POI search | Free | ~10,000 req/day |
| Wikivoyage | Travel guide text | Free | Reasonable use |

**Cost estimate:** A typical 3-day itinerary with 3–5 tool calls costs approximately $0.02–0.05 using GPT-4o, or under $0.005 with GPT-4o-mini.

---

## 📖 Example Use Cases

**1. Food-focused city trip**
- Destination: San Francisco, CA
- Interests: Food, Culture
- Constraints: ramen restaurants
- Result: 3-day itinerary with pinned ramen spots across the city

**2. Outdoor adventure**
- Destination: Denver, Colorado
- Interests: Outdoors, History
- Constraints: hiking trails near downtown
- Result: Trail recommendations with park locations on map

**3. Quick weekend preview (Fast Mode)**
- Toggle Fast Mode ON + select GPT-4o-mini
- Get a 2-day itinerary in under 15 seconds

**4. Refine an existing plan**
- Generate a base itinerary, then use "Refine Your Itinerary" to swap Day 2 dinner for a specific cuisine type

---

## ⚠️ Known Limitations

- **OSM data gaps** — Smaller cities (e.g. Kona, Hawaii) have sparse POI coverage; the app falls back to city-center pin
- **Ambiguous city names** — Use full names like "Kona, Hawaii" instead of just "Kona" to avoid geocoding to the wrong country
- **Feedback persistence** — The `poi_feedback.jsonl` file is local; feedback does not persist across Streamlit Cloud deployments without external storage
- **Agent hallucination** — The LLM occasionally fabricates coordinates (filtered out by the 0,0 validator) or invents venue names not in OSM
- **Rate limits** — Heavy concurrent usage may trigger Nominatim or Overpass rate limiting (429 responses); exponential backoff is implemented

---

## 🔮 Future Enhancements

- [ ] Google Places API integration for richer venue data
- [ ] User authentication and saved itineraries
- [ ] Export to PDF or Google Calendar
- [ ] Multi-city trip planning
- [ ] Hotel and flight integration
- [ ] Persistent feedback storage (PostgreSQL or Supabase)
- [ ] Budget estimation per day

---

## 🙏 Credits & Resources

- [OpenAI API](https://platform.openai.com/) — GPT-4o for agent orchestration
- [OpenStreetMap / Overpass API](https://overpass-api.de/) — POI data
- [Nominatim](https://nominatim.org/) — Geocoding
- [Wikivoyage](https://en.wikivoyage.org/) — Travel guide content
- [Streamlit](https://streamlit.io/) — App framework
- [PyDeck](https://deckgl.readthedocs.io/) — Map rendering
- [scikit-learn TF-IDF](https://scikit-learn.org/) — RAG retrieval

---

## 📄 License

MIT License — feel free to fork and build on this project.
