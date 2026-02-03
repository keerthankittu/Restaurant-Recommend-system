# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches

# ---------- Helpers ----------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9, ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

@st.cache_data(ttl=3600)
def load_and_prepare(csv_path='restaurant.csv'):
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist - try common names
    # Adjust if your column names are different
    if 'restaurant_name' not in df.columns:
        # try other guesses
        possible = [c for c in df.columns if 'name' in c.lower()]
        if possible:
            df = df.rename(columns={possible[0]:'restaurant_name'})
    if 'cuisines' not in df.columns:
        possible = [c for c in df.columns if 'cuisin' in c.lower() or 'categ' in c.lower()]
        if possible:
            df = df.rename(columns={possible[0]:'cuisines'})
    # Fill missing
    df['restaurant_name'] = df['restaurant_name'].fillna('').astype(str)
    df['cuisines'] = df.get('cuisines', '').fillna('').astype(str)
    # combined text
    df['combined_text'] = (df['restaurant_name'] + ' ' + df['cuisines']).apply(clean_text)
    # Display name (disambiguate duplicates)
    df['display_name'] = df['restaurant_name'] + ' - idx:' + df.index.astype(str)
    return df

@st.cache_data(ttl=3600)
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['display_name']).drop_duplicates()
    return tfidf, tfidf_matrix, cosine_sim, indices

def recommend(df, cosine_sim, indices, query_display_name, topn=5):
    if query_display_name not in indices:
        # try fuzzy search on restaurant_name
        names = df['restaurant_name'].tolist()
        matches = get_close_matches(query_display_name, names, n=5, cutoff=0.6)
        if matches:
            # return candidate suggestions
            return None, matches
        else:
            return None, []
    idx = indices[query_display_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:topn+1]  # skip itself
    restaurant_indices = [i[0] for i in sim_scores]
    results = df.iloc[restaurant_indices][['display_name','restaurant_name','cuisines']].copy()
    scores = [s for (_,s) in sim_scores]
    results['score'] = np.round(scores, 3)
    return results.reset_index(drop=True), None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Restaurant Recommender", layout="centered")
st.title("Content-based Restaurant Recommendation")
st.markdown("Type a restaurant name (or select from dropdown) and get top-N similar restaurants based on name + cuisine text.")

# Load data & model
csv_path = st.text_input("CSV path (leave blank for 'restaurant.csv')", value="restaurant.csv")
df = load_and_prepare(csv_path)
tfidf, tfidf_matrix, cosine_sim, indices = build_model(df)

# Input UI
st.subheader("Find similar restaurants")
# Give user a dropdown sample to choose from or type manually
sample_display = st.selectbox("Choose from sample restaurants (or type below):", options=df['display_name'].sample(min(50, len(df))).tolist())
user_input = st.text_input("Or type a restaurant name (case-insensitive):", value="")

topn = st.slider("Number of recommendations (Top N):", 1, 20, 5)

# Determine which display name to use
query_display_name = None
if user_input.strip() != "":
    # if they typed exact display_name with idx, use it; else try to match restaurant_name (fuzzy)
    typed = user_input.strip()
    if typed in df['display_name'].values:
        query_display_name = typed
    else:
        # simple best-name match via close matches
        close = get_close_matches(typed, df['restaurant_name'].tolist(), n=1, cutoff=0.6)
        if close:
            # get first matched restaurant's display_name
            idx = df[df['restaurant_name'] == close[0]].index[0]
            query_display_name = df.loc[idx, 'display_name']
        else:
            query_display_name = None
else:
    query_display_name = sample_display

if st.button("Get Recommendations"):
    results, suggestions = recommend(df, cosine_sim, indices, query_display_name, topn=topn)
    if results is None and suggestions is not None:
        if len(suggestions) == 0:
            st.warning("No match found. Try another name or choose from the dropdown.")
        else:
            st.info("No exact match — did you mean one of these restaurant names?")
            for s in suggestions:
                st.write("-", s)
    elif results is None:
        st.warning("No match found. Try a different name.")
    else:
        st.success(f"Top {topn} recommendations for: **{df.loc[indices[query_display_name], 'restaurant_name']}**")
        st.dataframe(results)

st.markdown("---")
st.write("Total restaurants in dataset:", len(df))
st.write("Example display names (you can copy one and paste above):")
st.write(df['display_name'].sample(min(10,len(df))).tolist())
