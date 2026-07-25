import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 5
RESULTS_PER_PAGE = 5  # must divide evenly into the API's N_RESULTS (25)

st.set_page_config(page_title="Restaurant Song Recommender")
st.title("Restaurant Song Recommender")


@st.cache_data(ttl=30)
def get_health():
    resp = requests.get(f"{API_BASE_URL}/health", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


try:
    health = get_health()
    st.caption(
        f"API: `{API_BASE_URL}` · artifact: `{health['artifact_file']}` "
        f"(built {health['built_at']}) · {health['n_songs']:,} songs · "
        f"cuisine filters: {'yes' if health['has_cuisine_filters'] else 'no'} · "
        f"popularity data: {'yes' if health['has_popularity'] else 'no'}"
    )
except requests.exceptions.RequestException as e:
    st.caption(f":warning: Couldn't reach API health check at {API_BASE_URL}: {e}")

for key in ("search_results", "selected_business_id", "recommendation", "recommend_key"):
    if key not in st.session_state:
        st.session_state[key] = None

if "recommend_round" not in st.session_state:
    st.session_state.recommend_round = 0

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}  # (business_id, song_id) -> "like" / "dislike"

if "popularity_boost" not in st.session_state:
    st.session_state.popularity_boost = 0.0


@st.cache_data(ttl=300)
def search_restaurants(query: str):
    resp = requests.get(
        f"{API_BASE_URL}/restaurants", params={"q": query, "limit": 10}, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def get_recommendation(business_id: str, popularity_boost: float = 0.0):
    resp = requests.post(
        f"{API_BASE_URL}/recommend",
        json={"business_id": business_id, "popularity_boost": popularity_boost},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_feedback(business_id: str, song_id: str, direction: str):
    resp = requests.post(
        f"{API_BASE_URL}/feedback",
        json={"business_id": business_id, "song_id": song_id, "direction": direction},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()


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
    st.session_state.popularity_boost = 0.0
    if not results:
        st.warning("No restaurants found for that search.")

results = st.session_state.search_results
if results:
    options = {r["name"]: r["business_id"] for r in results}
    choice_name = st.selectbox("Select a restaurant", list(options.keys()))
    st.session_state.selected_business_id = options[choice_name]

    if st.button("Recommend songs"):
        recommend_key = (st.session_state.selected_business_id, st.session_state.popularity_boost)
        try:
            if st.session_state.recommendation is None or st.session_state.recommend_key != recommend_key:
                st.session_state.recommendation = get_recommendation(
                    st.session_state.selected_business_id, st.session_state.popularity_boost
                )
                st.session_state.recommend_key = recommend_key
                st.session_state.recommend_round = 0
            else:
                # Same restaurant + boost as last click - page to the next batch of
                # already-fetched songs instead of re-querying. Wraps back to the
                # first batch after the last page rather than erroring.
                n_rounds = max(
                    1, len(st.session_state.recommendation["recommendations"]) // RESULTS_PER_PAGE
                )
                st.session_state.recommend_round = (st.session_state.recommend_round + 1) % n_rounds
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't reach the API at {API_BASE_URL}. Is it running? ({e})")

recommendation = st.session_state.recommendation
if recommendation:
    boost_pct = st.slider(
        "Prefer more well known songs",
        min_value=0,
        max_value=100,
        value=int(st.session_state.popularity_boost * 100),
        step=10,
        help="Not feeling these picks? Slide right to favor more popular songs.",
    )
    new_boost = boost_pct / 100
    if new_boost != st.session_state.popularity_boost:
        st.session_state.popularity_boost = new_boost
        try:
            st.session_state.recommendation = get_recommendation(
                recommendation["business_id"], new_boost
            )
            recommendation = st.session_state.recommendation
            st.session_state.recommend_key = (recommendation["business_id"], new_boost)
            st.session_state.recommend_round = 0
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't refresh recommendations: {e}")

    all_recs = recommendation["recommendations"]
    n_rounds = max(1, len(all_recs) // RESULTS_PER_PAGE)
    start = st.session_state.recommend_round * RESULTS_PER_PAGE
    page_recs = all_recs[start : start + RESULTS_PER_PAGE]

    st.subheader(f"Top 5 songs for {recommendation['restaurant_name']}")
    st.caption(f"Dominant vibe: {recommendation['dominant_spotify_label']}")

    matched_genres = recommendation.get("matched_cuisine_genres")
    if matched_genres:
        st.caption(f"Cuisine match, filtered to: {', '.join(matched_genres)}")

    for i, song in enumerate(page_recs, 1):
        note = " _(common pick across many restaurants)_" if song["is_hub"] else ""
        detail_bits = [b for b in (song.get("genre"), song.get("popularity")) if b is not None]
        detail = f"  ({', '.join(str(b) for b in detail_bits)})" if detail_bits else ""

        text_col, like_col, dislike_col = st.columns([8, 1, 1])
        text_col.write(f"{i}. **{song['name']}** by {song['artists']}{detail}{note}")

        feedback_key = (recommendation["business_id"], song["id"])
        given = st.session_state.feedback_given.get(feedback_key)
        if given:
            text_col.caption("You: 👍 liked" if given == "like" else "You: 👎 disliked")
        else:
            if like_col.button("👍", key=f"like_{i}_{song['id']}"):
                try:
                    send_feedback(recommendation["business_id"], song["id"], "like")
                    st.session_state.feedback_given[feedback_key] = "like"
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Couldn't record feedback: {e}")
            if dislike_col.button("👎", key=f"dislike_{i}_{song['id']}"):
                try:
                    send_feedback(recommendation["business_id"], song["id"], "dislike")
                    st.session_state.feedback_given[feedback_key] = "dislike"
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Couldn't record feedback: {e}")
