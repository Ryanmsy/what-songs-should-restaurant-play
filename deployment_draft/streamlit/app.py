import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 5

st.set_page_config(page_title="Restaurant Song Recommender")
st.title("Restaurant Song Recommender")

for key in ("search_results", "selected_business_id", "recommendation"):
    if key not in st.session_state:
        st.session_state[key] = None


@st.cache_data(ttl=300)
def search_restaurants(query: str):
    resp = requests.get(
        f"{API_BASE_URL}/restaurants", params={"q": query, "limit": 10}, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def get_recommendation(business_id: str):
    resp = requests.post(
        f"{API_BASE_URL}/recommend", json={"business_id": business_id}, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


query = st.text_input("Search for a restaurant by name")

if st.button("Search") and query.strip():
    try:
        results = search_restaurants(query.strip())
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the API at {API_BASE_URL}. Is it running? ({e})")
        results = []

    st.session_state.search_results = results
    st.session_state.selected_business_id = None
    st.session_state.recommendation = None
    if not results:
        st.warning("No restaurants found for that search.")

results = st.session_state.search_results
if results:
    options = {r["name"]: r["business_id"] for r in results}
    choice_name = st.selectbox("Select a restaurant", list(options.keys()))
    st.session_state.selected_business_id = options[choice_name]

    if st.button("Recommend songs"):
        try:
            st.session_state.recommendation = get_recommendation(st.session_state.selected_business_id)
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't reach the API at {API_BASE_URL}. Is it running? ({e})")

recommendation = st.session_state.recommendation
if recommendation:
    st.subheader(f"Top 5 songs for {recommendation['restaurant_name']}")
    st.caption(f"Dominant vibe: {recommendation['dominant_spotify_label']}")
    for i, song in enumerate(recommendation["recommendations"], 1):
        note = " _(common pick across many restaurants)_" if song["is_hub"] else ""
        st.write(f"{i}. **{song['name']}** by {song['artists']}{note}")
