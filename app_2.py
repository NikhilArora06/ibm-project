import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

st.set_page_config(page_title="Netflix Notebook Results", layout="wide")
st.title("🎬 Netflix EDA — Reproduced from Your Notebook")

# ---------- Data loading ----------
st.sidebar.header("Data")
sample_path = Path(__file__).parent / "netflix_titles.csv"
use_sample = sample_path.exists()
uploaded = st.sidebar.file_uploader("Upload netflix_titles.csv", type=["csv"])

@st.cache_data
def load_csv(path_or_buf):
    return pd.read_csv(path_or_buf)

if uploaded is not None:
    df = load_csv(uploaded)
elif use_sample:
    st.sidebar.info("Using included sample `netflix_titles.csv`.")
    df = load_csv(sample_path)
else:
    st.info("Upload your CSV to view results.")
    st.stop()

raw_df = df.copy()

# ---------- Notebook-like cleaning steps ----------
df = df.drop_duplicates()
nulls = df.isnull().sum().sort_values(ascending=False)

if "date_added" in df.columns:
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["added_year"] = df["date_added"].dt.year
    df["added_month"] = df["date_added"].dt.month
    df["added_dayofweek"] = df["date_added"].dt.dayofweek

def parse_duration(d):
    if pd.isna(d):
        return np.nan, np.nan
    s = str(d).lower()
    if "min" in s:
        try:
            return float(s.split()[0]), np.nan
        except:
            return np.nan, np.nan
    if "season" in s:
        try:
            return np.nan, float(s.split()[0])
        except:
            return np.nan, np.nan
    return np.nan, np.nan

if "duration" in df.columns:
    minutes, seasons = zip(*df["duration"].map(parse_duration))
    df["duration_minutes"] = minutes
    df["seasons"] = seasons

if "rating" in df.columns:
    df = df[~df["rating"].astype(str).str.contains("min", na=False)]

# ---------- Layout ----------
st.subheader("🔎 Dataset preview")
st.dataframe(raw_df.head(10), use_container_width=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(df))
c2.metric("Columns", df.shape[1])
c3.metric("Movies %", f"{(df['type'].eq('Movie').mean()*100):.1f}%" if 'type' in df.columns else "–")
c4.metric("Unique Countries", df['country'].dropna().str.split(',').explode().str.strip().nunique() if 'country' in df.columns else 0)

with st.expander("🧰 Missing values (top 10)"):
    st.table(nulls.head(10))

st.markdown("---")

# ---------- Global plot style (to match your notebook) ----------
plt.style.use('dark_background')
sns.set_palette('terrain_r')
sns.set_style("darkgrid")

# ---------- Plots from notebook (extended) ----------

# 1) Type distribution
if "type" in df.columns:
    st.subheader("Type distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x='type', hue='type', palette='terrain_r', legend=False, ax=ax)
    ax.set_title("Movies vs TV Shows on Netflix")
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# 2) Top 10 Genres
if "listed_in" in df.columns:
    st.subheader("Top 10 Genres on Netflix")
    genre_list = df['listed_in'].dropna().apply(lambda x: x.split(', '))
    all_genres = [genre for sublist in genre_list for genre in sublist]
    genre_counts = Counter(all_genres).most_common(10)
    genres_df = pd.DataFrame(genre_counts, columns=['Genre', 'Count'])
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=genres_df, x='Count', y='Genre', hue='Genre', palette='terrain', legend=False, ax=ax)
    ax.set_title("Top 10 Genres on Netflix")
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Genre")
    st.pyplot(fig)

# 3) Top 10 Countries
if "country" in df.columns:
    st.subheader("Top 10 Producing Countries")
    country_list = df['country'].dropna().apply(lambda x: x.split(', '))
    all_countries = [country for sublist in country_list for country in sublist]
    country_counts = Counter(all_countries).most_common(10)
    countries_df = pd.DataFrame(country_counts, columns=['Country','Count'])
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=countries_df, x='Count', y='Country', hue='Country', palette='terrain', legend=False, ax=ax)
    ax.set_title("Top 10 Countries Producing Netflix Content")
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Country")
    st.pyplot(fig)

# 4) Distribution of Content Ratings
if "rating" in df.columns:
    st.subheader("Distribution of Content Ratings on Netflix")
    rating_counts = df['rating'].value_counts().reset_index()
    rating_counts.columns = ['Rating','Count']
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=rating_counts, x='Count', y='Rating', hue='Rating', palette='terrain', legend=False, ax=ax)
    ax.set_title("Distribution of Content Ratings on Netflix")
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

# 5) Top 10 Directors
if "director" in df.columns:
    st.subheader("Top 10 Directors on Netflix")
    top_directors = (df['director']
                     .dropna()
                     .apply(lambda x: x.split(', '))
                     .explode()
                     .value_counts()
                     .head(10)
                     .reset_index())
    top_directors.columns = ['Director','Count']
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=top_directors, x='Count', y='Director', hue='Director', palette='terrain', legend=False, ax=ax)
    ax.set_title("Top 10 Directors on Netflix")
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Director")
    st.pyplot(fig)

# 6) Top 10 Actors
if "cast" in df.columns:
    st.subheader("Top 10 Actors Appearing on Netflix")
    top_actors = (df['cast']
                  .dropna()
                  .apply(lambda x: x.split(', '))
                  .explode()
                  .value_counts()
                  .head(10)
                  .reset_index())
    top_actors.columns = ['Actor','Count']
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=top_actors, x='Count', y='Actor', hue='Actor', palette='terrain', legend=False, ax=ax)
    ax.set_title("Top 10 Actors Appearing on Netflix")
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Actor")
    st.pyplot(fig)

# 7) Movie & TV durations
if all(col in df.columns for col in ("type", "duration")):
    st.subheader("Duration Analysis")
    movies = df[df['type'] == 'Movie'].copy()
    tv_shows = df[df['type'] == 'TV Show'].copy()
    # Movies
    movies['duration_int'] = movies['duration'].str.replace(' min', '').astype(float)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(movies['duration_int'].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Movie Durations on Netflix")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Number of Movies")
    st.pyplot(fig)
    # TV Shows (number of seasons)
    tv_shows['duration_int'] = tv_shows['duration'].str.replace(' Season','').str.replace('s','').astype(float)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.countplot(x=tv_shows['duration_int'].dropna(), palette='terrain', ax=ax)
    ax.set_title("Number of Seasons in Netflix TV Shows")
    ax.set_xlabel("Seasons")
    ax.set_ylabel("Number of Shows")
    st.pyplot(fig)

# 8) Heatmap: Rating vs Type
if all(col in df.columns for col in ("rating","type","show_id")):
    st.subheader("Rating × Type heatmap")
    pivot = df.pivot_table(index="rating", columns="type", values="show_id", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3*len(pivot))))
    sns.heatmap(pivot, annot=True, fmt="d", cmap='terrain_r', ax=ax)
    ax.set_ylabel("Rating")
    ax.set_xlabel("Type")
    st.pyplot(fig)

st.markdown("---")