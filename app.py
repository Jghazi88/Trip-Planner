import streamlit as st
import requests
import openai
from openai import OpenAI
import json
import re
import time
import os
import pandas as pd
import pydeck as pdk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit import elements

# --- Feedback System ---

FEEDBACK_FILE = os.path.join(os.path.expanduser("~"), "poi_feedback.jsonl")

def log_feedback(city_key, poi_id, vote):
    """Stores a feedback event in a JSONL file."""
    event = {
        "ts": time.time(),
        "city_key": city_key.lower(),
        "poi_id": str(poi_id),
        "vote": vote
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

def feedback_boost_map(city_key):
    """Calculates cumulative boost scores for a specific city."""
    boosts = {}
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("city_key") == city_key.lower():
                        pid = data["poi_id"]
                        change = 0.25 if data["vote"] == "up" else -0.35
                        boosts[pid] = boosts.get(pid, 0.0) + change
                except json.JSONDecodeError:
                    continue
    return boosts

# --- Helper Functions ---

def extract_itinerary_json(text):
    """Safely extracts and parses JSON from the agent's response."""
    if not text: 
        return None
    
    text = text.strip()
    data = None 

    try:
        # Attempt 1: If it's already pure JSON 
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt 2: Fallback to regex if the model used markdown backticks
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                return None

    return data

def clean_html(raw_html):
    """Remove HTML tags for the RAG system"""
    cleanr = re.compile('<.*?>')
    return " ".join(re.sub(cleanr, '', raw_html).split())

def get_weather(city_name):
    """Fetch current weather for a city using OpenWeatherMap free tier."""
    weather_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not weather_key:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_name, "appid": weather_key, "units": "imperial"}
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            d = res.json()
            return {
                "temp": round(d["main"]["temp"]),
                "feels_like": round(d["main"]["feels_like"]),
                "description": d["weather"][0]["description"].capitalize(),
                "humidity": d["main"]["humidity"],
                "icon": d["weather"][0]["icon"]
            }
    except:
        return None


# --- 1. Page Config ---
st.set_page_config(page_title="Agentic Trip Planner", layout="wide")
st.title("🌍 Agentic Trip Planner")

# --- 2. API Key & Session State ---
with st.sidebar:
    st.header("Configuration")
    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = ""
    if "map_points" not in st.session_state:
        st.session_state.map_points = []
    if "itinerary" not in st.session_state:
        st.session_state.itinerary = None
    if "trace" not in st.session_state:
        st.session_state.trace = []
    if "itinerary_json" not in st.session_state:
        st.session_state.itinerary_json = None
    if "poi_cache" not in st.session_state:
        st.session_state.poi_cache = {}
    if "compare_json" not in st.session_state:
        st.session_state.compare_json = None

    default_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input("Enter OpenAI API Key", type="password", value=st.session_state['openai_api_key'] or default_key)
    if api_key:
        st.session_state['openai_api_key'] = api_key

    st.divider()
    st.subheader("⚙️ Agent Settings")

    model_choice = st.selectbox("Model",
                                ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                                help="gpt-4o-mini is faster and cheaper; gpt-4o gives richer itineries."
                                ) 
    fast_mode = st.toggle(
        "⚡ Fast Mode",
        value=False,
        help="Limits tool calls for quicker results. Best for quick trip previews."
    )

    max_steps = st.slider(
        "Max Agent Steps",
        min_value=3, max_value=12,
        value=5 if fast_mode else 8,
        help="More steps = more thorough research but slower response."
    )

    st.caption("💡 Tip: Use Fast Mode + gpt-4o-mini for quick previews, then switch to gpt-4o for final planning.")

# --- 3. Connection Testing Logic ---
st.header("System Connectivity Check")

def test_osm():
    # ✅ Fix — more specific User-Agent that Nominatim accepts
    headers = {'User-Agent': 'CapstoneTripPlanner/1.0 (github.com/Jghazi88/Trip-Planner)'}
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geocoding (Nominatim)")
        try:
            res = requests.get("https://nominatim.openstreetmap.org/search?format=json&q=San+Jose", headers=headers, timeout=10)
            if res.status_code == 200: st.success("✅ Nominatim Connected")
            else: st.error(f"❌ Error: {res.status_code}")
        except Exception as e: st.error(f"❌ Connection Failed: {e}")
    with col2:
        st.subheader("POI Search (Overpass)")
        try:
            res = requests.get("https://overpass-api.de/api/interpreter?data=[out:json];node(50.746,7.154,50.748,7.157);out;", headers=headers, timeout=10)
            if res.status_code == 200: st.success("✅ Overpass Connected")
            else: st.error(f"❌ Error: {res.status_code}")
        except Exception as e: st.error(f"❌ Connection Failed: {e}")

if st.button("Run Diagnostic Tests"):
    test_osm()

st.divider()

# --- Task 2 & 8: Core Geocoding and POI Search Logic ---

INTEREST_MAP = {
    "food": ["restaurant", "cafe", "food_court"],
    "history": ["museum", "monument", "memorial"],
    "outdoors": ["park", "garden", "nature_reserve"],
    "culture": ["arts_centre", "theatre", "gallery"],
    "shopping": ["mall", "marketplace"]
}

@st.cache_data(ttl=3600)
def get_coordinates(city_name):
    if not city_name.strip():
        return None, None
    
    headers = {'User-Agent': 'CapstoneTripPlanner/1.0 (github.com/Jghazi88/Trip-Planner)'}
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={city_name}"

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            response.raise_for_status()
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
            return None, None
        except Exception as e:
            if attempt == 2: break
            time.sleep(2 ** attempt)
    return None, None

@st.cache_data(ttl=3600)
def search_pois(lat, lon, interest, query=None, city_key=""):
    headers = {'User-Agent': 'CapstoneTripPlanner/1.0 (github.com/Jghazi88/Trip-Planner)'}
    # ✅ Replace the query branch in search_pois
    if query:
        overpass_query = (
            f'[out:json];('
            f'node["name"~"{query}",i](around:15000,{lat},{lon});'        # name match
            f'node["cuisine"~"{query}",i](around:15000,{lat},{lon});'     # cuisine match
            f'way["cuisine"~"{query}",i](around:15000,{lat},{lon});'      # cuisine on ways too
            f');out center;'
        )
    else:
        amenity_tags = ["restaurant", "cafe", "food_court", "museum", "theatre", 
                    "arts_centre", "gallery", "monument", "memorial"]
        leisure_tags = ["park", "garden", "nature_reserve"]
        shop_tags = ["mall", "marketplace"]
    
        tags = INTEREST_MAP.get(interest, ["restaurant"])
        filters = ""
        for tag in tags:
            if tag in leisure_tags:
                filters += f'node["leisure"="{tag}"](around:15000,{lat},{lon});'
            elif tag in shop_tags:
                filters += f'node["shop"="{tag}"](around:15000,{lat},{lon});'
            else:
                filters += f'node["amenity"="{tag}"](around:15000,{lat},{lon});'
    
        overpass_query = f"[out:json];({filters});out center;"

    for attempt in range(3):
        try:
            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={'data': overpass_query},
                headers=headers,
                timeout=15
            )
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()

            elements = response.json().get('elements', [])
            print(f"DEBUG: Overpass returned {len(elements)} elements for interest={interest}, query={query}")
            if not elements:
                return []

            boosts = feedback_boost_map(city_key)
            pois = []
            for e in elements:
                lat = e.get('lat') or e.get('center', {}).get('lat')
                lon = e.get('lon') or e.get('center', {}).get('lon')
                if not lat or not lon:
                    continue
                poi_id = str(e.get('id', ''))
                poi = {
                    "poi_id": poi_id,
                    "name": e.get('tags', {}).get('name', 'Unnamed'),
                    "lat": lat,
                    "lon": lon,
                    "_base_score": 0.0
                }
                poi["_score"] = poi["_base_score"] + boosts.get(poi_id, 0.0)
                pois.append(poi)
                pois.sort(key=lambda x: x["_score"], reverse=True)
                return pois
        except Exception as e:
            if attempt == 2:
                return []
            time.sleep(2 ** attempt)
    return []
# --- Task 2: UI Implementation ---
col_city, col_interest = st.columns(2)
with col_city:
    city_input = st.text_input("Target City", value="", key="manual_city")
with col_interest:
    interest_input = st.selectbox("Interest Category", list(INTEREST_MAP.keys()), key="manual_interest")

if st.button("Execute search_pois Tool"):
    with st.spinner("Agent is fetching data..."):
        lat, lon = get_coordinates(city_input)
        if lat and lon:
            results = search_pois(lat, lon, interest=interest_input)
            if results:
                st.success(f"Found {len(results)} locations.")
                st.table(results[:10])
            else: st.warning("No POIs found.")
        else: st.error("Geocoding failed.")

# --- Task 3: Wikivoyage RAG System ---

@st.cache_data(ttl=3600)
def fetch_wikivoyage_data(city_name):
    url = "https://en.wikivoyage.org/w/api.php"
    params = {"action": "query", "prop": "extracts", "titles": city_name, "format": "json", "explaintext": True, "redirects": 1, "formatversion": 2}
    try:
        res = requests.get(url, params=params,headers = {'User-Agent': 'CapstoneTripPlanner/1.0 (github.com/Jghazi88/Trip-Planner)'}, timeout=10)
        return res.json().get("query", {}).get("pages", [{}])[0].get("extract", "")
    except: return ""

def retrieve_context(query, text, k=3):
    if not text: return []
    chunks = [text[i:i + 900] for i in range(0, len(text), 900)]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return [{"chunk_id": idx, "text": chunks[idx], "score": round(float(scores[idx]), 3)} for idx in top_indices]

# --- Task 4: AI Agent Orchestration ---

def run_agent_loop(messages, tools, client, max_steps=6, model="gpt-4o", fast_mode=False):  
    execution_log = []

    loop_start = time.time()

    # Fast mode vs full mode instructions
    if fast_mode:
        mode_instruction = " Call search_pois ONCE  with broad criteria."
    else:
        mode_instruction = " Call search_pois 2-3 times with different queries for thorough coverage."
    
    # Inject mode instruction into the last user message
    messages[-1]["content"] += mode_instruction

    for i in range(max_steps):
        step_start = time.time()
        try:
            response = client.chat.completions.create(
                model=model, 
                messages=messages, 
                tools=tools, 
                timeout=45,
                response_format={"type": "json_object"}
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            step_time = round(time.time() - step_start, 2)

            if not msg.tool_calls:
                total_time = round(time.time() - loop_start, 2)
                execution_log.append(f"✅ Final response in {step_time}s (total: {total_time}s)")
                return msg.content, execution_log

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                call_id = tool_call.id
                
                try:
                    args = json.loads(tool_call.function.arguments)
                    execution_log.append(f"Tool: {fn_name}({args})")
                    
                    result = {"error": "Tool execution failed"}
                    
                    if fn_name == "search_pois":
                        target_city = args.get('city', '')
                        lat, lon = get_coordinates(target_city)
                        if lat and lon:
                            poi_list = search_pois(lat, lon, interest=args.get('interest'), query=args.get('query'), city_key=target_city)
                            for p in poi_list:
                                st.session_state.poi_cache[str(p.get('poi_id'))] = p
                            
                            if poi_list:
                                for p in poi_list[:5]:
                                    entry=p.copy()
                                    entry['day'] = i + 1
                                    st.session_state.map_points.append(entry)

                            result = {"results": poi_list}
                        else:
                            result = {"error": f"Could not find coordinates for {target_city}"}

                    elif fn_name == "retrieve_guides":
                        raw = fetch_wikivoyage_data(args.get('city', ''))
                        context = retrieve_context(args.get('query', ''), clean_html(raw), k=1)
                        result = {"context": context[0]['text'] if context else "No info found."}

                except Exception as e:
                    result = {"error": str(e)}
                
                messages.append({
                    "role": "tool", 
                    "tool_call_id": call_id, 
                    "name": fn_name, 
                    "content": json.dumps(result)
                })

        except openai.APITimeoutError:
            return "The AI took too long to respond.", execution_log
        except openai.APIError as e:
            st.error(f"OpenAI API Error: {e}")
            return "API Error occurred.", execution_log
        except Exception as e:
            st.error(f"Unexpected Error: {e}")
            return "Agent loop crashed.", execution_log
            
    return "Steps exceeded max limit.", execution_log

# --- Task 4 & 6: Map Rendering Logic ---

def render_itinerary_map(points_list, selected_day="All"):
    if not points_list or not isinstance(points_list, list):
        return

    df = pd.DataFrame(points_list)
    
    if 'day' not in df.columns:
        st.warning("Data is missing 'day' labels. Showing all points.")
        selected_day = "All"
    
    if selected_day != "All":
        df = df[df['day'].astype(str) == str(selected_day)]

    if df.empty:
        st.warning(f"No valid coordinates to show for Day {selected_day}.")
        return
    
    # Dynamic dot size -- fewer points = larger dots for visibility
    dot_radius = max(80, 300 - (len(df) * 10))  # Base size minus 10 for each point, with a minimum of 80

    view = pdk.ViewState(latitude=df["lat"].mean(), longitude=df["lon"].mean(), zoom=10)
    layer = pdk.Layer("ScatterplotLayer", data=df, get_position="[lon, lat]", get_color="[200, 30, 0, 160]", get_radius=dot_radius, pickable=True, auto_highlight=True)
    st.pydeck_chart(pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json", layers=[layer], initial_view_state=view, tooltip={"text": "{name}\nDay: {day}"}))

# --- Task 5: UI & Generation ---

st.header("✈️ AI Travel Designer Pro")
with st.expander("📝 Trip Specifications", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        dest = st.text_input("Destination", "San Jose", help="Enter a city name. Be specific (e.g. 'Kona, Hawaii') to avoid ambiguity")
        days = st.slider("Trip Length (Days)", 1, 7, 3, help="Number of days for the trip, not including flights")
    with col2:
        pace = st.selectbox("Pace", ["Relaxed", "Balanced", "Fast-paced"], help="Relaxed = fewer stops, more leisure. Fast-paced = maximize sights and travel")
        interests = st.multiselect("Key Interests", list(INTEREST_MAP.keys()), default=["food"], help="Select one or more categories to maximize itinerary.")
    constraints = st.text_input("Constraints", placeholder="e.g. sushi restaurants, hiking trails, no museums",
                                help="Optional: specific venues, cuisine types, or things to avoid.")

tools = [
    {
        "type": "function", 
        "function": {
            "name": "search_pois", 
            "description": "Find places by general category.", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "city": {"type": "string"}, 
                    "interest": {
                        "type": "string", 
                        "enum": list(INTEREST_MAP.keys())
                    }, 
                    "query": {
                        "type": "string", 
                        "description": "Optional: A place name OR cuisine type to search for (e.g., 'Starbucks', 'sushi', 'Mexican'). Use this when the user requests a specific type of food or a named venue."
                    }
                }, 
                "required": ["city", "interest"], 
                "additionalProperties": False
            }
        }
    },
    {"type": "function", "function": {"name": "retrieve_guides", "description": "Read guides.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "query": {"type": "string"}}, "required": ["city", "query"], "additionalProperties": False}, "strict": True}}
]

if st.button("🚀 Generate Itinerary"):
    if not dest.strip():
        st.error("Please enter a destination city.")
        st.stop()
    
    if not interests:
        st.warning("Please select at least one interest to help the agent.")
        st.stop()

    # Pre-validation step for robustness 
    lat_check, lon_check = get_coordinates(dest)
    if lat_check is None:
        st.error(f"Geocoding failed: Could not find coordinates for '{dest}'. Please check the spelling or try a different city.")
        st.stop()

    st.session_state.map_points = []
    st.session_state.poi_cache = {} 

    with st.status("🤖 Agent at work...", expanded=True) as status:
        client = OpenAI(api_key=st.session_state['openai_api_key'])
        # ✅ Fix — include constraints in the prompt
        # ✅ Add + before the conditional line
        prompt = (            
               
            f"Plan a {days}-day trip to {dest}. Interests: {', '.join(interests)}. "
            + (f"User constraints and special requests: {constraints}. " if constraints.strip() else "")
            + "Use the search_pois tool to find real venues. "
            "IMPORTANT: When search_pois returns results, use the EXACT 'poi_id', 'name', 'lat', and 'lon' "
            "values from the tool results in your locations list. Do NOT invent new values. "
            "CRITICAL: You must return a SINGLE JSON object. Do not include any text outside the JSON block. "
            "The JSON must have these keys:\n"
            "1. 'itinerary_text': Your full travel guide formatted in Markdown.\n"
            "2. 'locations': A list of POIs with 'name', 'lat', 'lon', 'poi_id', and 'day'.\n\n"
            "If no POIs are found for a category, include the advice in 'itinerary_text' but "
            "leave 'locations' empty for that item. Do not use 0.0 for coordinates."
        )

        search_pois.clear()

        ans, logs = run_agent_loop([{"role": "user", "content": prompt}], tools, client, max_steps=max_steps, model=model_choice, fast_mode=fast_mode)

        try:
            itinerary_data = extract_itinerary_json(ans)
            if itinerary_data is None:
                raise ValueError("The AI response did not contain a valid JSON itinerary block.")
        
            st.session_state.itinerary = ans
            st.session_state.trace = logs
            st.session_state.itinerary_json = itinerary_data
    
        except ValueError as e:
            st.error(f"❌ Could not parse itinerary: {e}")
            with st.expander("Raw model output"):
                st.code(ans)
            st.stop()
    st.rerun()

if st.button("🔄 Generate Alternative Itinerary (Compare)"):
    if not dest.strip() or not interests:
        st.warning("Please fill in destination and interests first.")
        st.stop()
    lat_check, lon_check = get_coordinates(dest)
    if lat_check is None:
        st.error("Geocoding failed for comparison.")
        st.stop()
    with st.status("🔄 Generating alternative itinerary...", expanded=True):
        client = OpenAI(api_key=st.session_state['openai_api_key'])
        compare_prompt = (            
            f"Plan a {days}-day trip to {dest}. Interests: {', '.join(interests)}. "
            + (f"User constraints and special requests: {constraints}. " if constraints.strip() else "")
            + "Use the search_pois tool to find real venues. "
            "IMPORTANT: When search_pois returns results, use the EXACT 'poi_id', 'name', 'lat', and 'lon' "
            "values from the tool results in your locations list. Do NOT invent new values. "
            "CRITICAL: You must return a SINGLE JSON object. Do not include any text outside the JSON block. "
            "The JSON must have these keys:\n"
            "1. 'itinerary_text': Your full travel guide formatted in Markdown.\n"
            "2. 'locations': A list of POIs with 'name', 'lat', 'lon', 'poi_id', and 'day'.\n\n"
            "If no POIs are found for a category, include the advice in 'itinerary_text' but "
            "leave 'locations' empty for that item. Do not use 0.0 for coordinates."
        )
        alt_ans, _ = run_agent_loop(
            [{"role": "user", "content": compare_prompt}],
            tools, client,
            max_steps=max_steps,
            model=model_choice,
            fast_mode=True
        )
        st.session_state.compare_json = extract_itinerary_json(alt_ans)
    st.rerun()

# --- Results & Refinement & Feedback ---

if st.session_state.itinerary_json is not None:
    st.divider()
    
    data = st.session_state.itinerary_json
    final_points = []
    # Catch the empty dictionary hallucination
    if not data:
        st.warning("The AI returned an empty response after checking locations. Please click 'Generate Itinerary' to try again.")
    else:
        display_text = data.get("itinerary_text", "No description available.")
        raw_points = data.get("locations", [])

        # --- ROBUST MAP FILTERING ---
        day_options = ["All"] + [str(i+1) for i in range(days)]
        day_filter = st.selectbox("Filter Map by Day", day_options)
    
        
        for pt in raw_points:
            poi_id = str(pt.get("poi_id", ""))
            if poi_id and poi_id in st.session_state.get("poi_cache", {}):
                pt["lat"] = st.session_state.poi_cache[poi_id]["lat"]
                pt["lon"] = st.session_state.poi_cache[poi_id]["lon"]
            else:
                name_match = next(
                    (p for p in st.session_state.get("map_points", []) if p.get("name") == pt.get("name")),
                    None
                )
                if name_match:
                    pt["lat"] = name_match["lat"]
                    pt["lon"] = name_match["lon"]
            try:
                lat = float(pt.get("lat", 0))
                lon = float(pt.get("lon", 0))
                pt["lat"] = lat
                pt["lon"] = lon
            except (ValueError, TypeError):
                continue

            if pt["lat"] != 0.0 and pt["lon"] != 0.0:
                final_points.append(pt)

        if not final_points:
        # Fallback 1: use real Overpass points saved during agent loop
            if st.session_state.get("map_points"):
                final_points = st.session_state.map_points
            else:
            # Fallback 2: pin the city center so map always renders something
                lat_c, lon_c = get_coordinates(dest)
                if lat_c:
                    final_points = [{"name": dest, "lat": lat_c, "lon": lon_c, "day": 1}]

        render_itinerary_map(final_points, day_filter)

        st.markdown(display_text)

        weather = get_weather(dest)
        if weather:
            st.divider()
            icon_url = f"https://openweathermap.org/img/wn/{weather['icon']}@2x.png"
            w_col1, w_col2, w_col3, w_col4 = st.columns(4)
            with w_col1:
                st.image(icon_url, width=60)
                st.caption(weather["description"])
            with w_col2:
                st.metric("🌡️ Temperature", f"{weather['temp']}°F")
            with w_col3:
                st.metric("🤔 Feels Like", f"{weather['feels_like']}°F")
            with w_col4:
                st.metric("💧 Humidity", f"{weather['humidity']}%")
            st.divider()

        # --- SIDE-BY-SIDE COMPARISON ---
    if st.session_state.compare_json:
        st.divider()
        st.subheader("🆚 Itinerary Comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Option A (Original)**")
            st.markdown(data.get("itinerary_text", ""))
        with col_b:
            st.markdown("**Option B (Alternative)**")
            st.markdown(st.session_state.compare_json.get("itinerary_text", ""))

        if st.button("✅ Keep Option B as My Itinerary"):
            st.session_state.itinerary_json = st.session_state.compare_json
            st.session_state.compare_json = None
            st.rerun()
        if  st.button("❌ Discard Comparison"):
            st.session_state.compare_json = None
            st.rerun()      


    # --- Task 8: User Feedback UI ---
    st.divider()
    st.subheader("⭐ Rate Your Locations")
    st.markdown("Help the AI learn! Upvote locations you loved, and downvote ones that missed the mark.")
    
    if final_points:
        for idx, pt in enumerate(final_points):
            name = pt.get("name", "Unknown Location")
            poi_id = pt.get("poi_id", str(name)) 
            
            col_name, col_up, col_down, _ = st.columns([4, 1, 1, 4])
            with col_name:
                st.write(f"**{name}** (Day {pt.get('day', '?')})")
            with col_up:
                if st.button("👍", key=f"up_{poi_id}_{idx}"):
                    log_feedback(dest, poi_id, "up")
                    st.success(f"Upvoted: {name}")
            with col_down:
                if st.button("👎", key=f"down_{poi_id}_{idx}"):
                    log_feedback(dest, poi_id, "down")
                    st.warning(f"Downvoted: {name}")
    else:
        st.info("Generate an itinerary first to rate locations.")

    # --- Robust Refinement Tool ---
    st.subheader("🛠️ Refine Your Itinerary")
    
    col_scope, col_req = st.columns([1, 3])
    with col_scope:
        scope = st.selectbox("Scope of Change", ["Entire Trip"] + [f"Day {i+1}" for i in range(days)])
    with col_req:
        refine_req = st.text_input("What would you like to change?", placeholder="e.g., Add a sushi place for dinner")

    if st.button("✨ Apply Changes"):
        if not refine_req:
            st.warning("Please enter a request.")
        else:
            client = OpenAI(api_key=st.session_state['openai_api_key'])
            
            # ✅ Fix — include the existing itinerary so the LLM can actually modify it
            existing_itinerary = data.get("itinerary_text", "") if st.session_state.itinerary_json else ""
            existing_locations = json.dumps(
                st.session_state.itinerary_json.get("locations", []) if st.session_state.itinerary_json else []
            )

            # ✅ Fix — include refine_req and scope
            refine_prompt = (
                f"Here is the existing {days}-day itinerary for {dest}:\n\n{existing_itinerary}\n\n"
                f"Here are the existing map locations (preserve these unless the scope requires changing them):\n"
                f"{existing_locations}\n\n"
                f"Please modify it based on this request: '{refine_req}'. "  # ← tells LLM what to change
                f"Scope of changes: {scope}. "                               # ← tells LLM the scope
                + (f"Remember the original constraints: {constraints}. " if constraints.strip() else "")
                + "Use the search_pois tool to find real venues for any new places requested. "
                + "IMPORTANT: Use EXACT 'poi_id', 'name', 'lat', 'lon' values from tool results. Do NOT invent new values. "
                + "CRITICAL: Return a SINGLE JSON object with keys:\n"
                + "1. 'itinerary_text': Full travel guide in Markdown.\n"
                + "2. 'locations': ALL POIs across all days with 'name', 'lat', 'lon', 'poi_id', 'day'. "
                + "Keep all existing locations unless the scope explicitly removes them.\n\n"
                + "Do not use 0.0 for coordinates."
            )
            
            with st.status("🔄 Updating your trip..."):
                
                new_ans, new_logs = run_agent_loop(
                    [{"role": "user", "content": refine_prompt}], 
                    tools, 
                    client,
                    max_steps=max_steps,
                    model=model_choice,
                    fast_mode=fast_mode
                )
                
                st.session_state.itinerary = new_ans
                st.session_state.itinerary_json = extract_itinerary_json(new_ans)
                st.session_state.trace = new_logs
                st.rerun()

if st.session_state.trace:
    with st.expander("🕵️ View Agent Logic & Timings"):
        st.caption("Each step shows what tool the agent called and how long it took.")
        for log in st.session_state.trace:
            if log.startswith("✅"):
                st.success(log)
            elif log.startswith("🔧"):
                st.code(log)
            else:
                st.text(log)
