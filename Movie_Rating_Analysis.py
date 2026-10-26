"""
MOVIE RATING ANALYSIS USING PYTHON
==================================

PROJECT OVERVIEW
----------------
This beginner-friendly project analyzes movie ratings using Python.
It demonstrates data cleaning, exploratory data analysis (EDA), summary
statistics, genre comparison, and charts.

TECHNOLOGIES
------------
- Python
- Pandas
- Matplotlib
- Seaborn

PROJECT OBJECTIVES
------------------
1. Find the average, highest, and lowest movie ratings.
2. Display the top-rated movies.
3. Compare average ratings across genres.
4. Clean missing or invalid values.
5. Save summary reports and charts.

INSTALLATION
------------
Open a terminal and run:
    pip install pandas matplotlib seaborn

HOW TO RUN
----------
Save this file as:
    movie_rating_analysis.py

Then run:
    python movie_rating_analysis.py

DATASET
-------
The program uses a built-in sample dataset by default, so it can run
without downloading another file. The sample data is only for learning
and demonstration. For a real portfolio project, replace it with a
larger, documented dataset and mention its source.

OPTIONAL CSV DATA
-----------------
You may place a file named "movies.csv" in the same folder as this script.
It should have columns for title, genre, and rating. Optional columns:
votes and year.

Example:
title,genre,rating,votes,year
Example Movie,Drama,8.1,25000,2020
Another Movie,Comedy,7.4,12000,2019

Common alternative column names are recognized, including:
movie_title, genres, imdb_rating, vote_count, release_year.

OUTPUTS
-------
The script creates an "outputs" folder containing:
- cleaned_movies.csv
- genre_summary.csv
- top_rated_movies.png
- average_rating_by_genre.png

RESUME DESCRIPTION
------------------
Movie Rating Analysis Using Python
- Cleaned and analyzed movie rating data using Pandas.
- Compared average ratings across genres and identified top-rated movies.
- Created visualizations using Matplotlib and Seaborn.
- Exported cleaned data and summary reports to CSV.

Only include resume statements that you have run, verified, and can
explain in an interview.

INTERVIEW DEFINITIONS
---------------------
Database/Data Analysis: Examining data to find useful information.
Pandas: A Python library for working with tabular data.
Data cleaning: Fixing or removing missing, invalid, or inconsistent data.
EDA: Exploring data using summaries and visualizations.
Average rating: Sum of ratings divided by the number of valid ratings.
Genre: A category of movie, such as Drama, Comedy, or Action.
Visualization: Showing data in charts so patterns are easier to understand.
CSV: A text file format that stores tabular data separated by commas.

IMPORTANT LIMITATION
--------------------
The built-in sample dataset is small and illustrative. Its results should
not be presented as findings from a large or official movie database.
If you use an external dataset, add the source and its license here.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Output files are saved in an "outputs" folder beside this script.
PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "movies.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def make_sample_data():
    """Create sample movie data so the project runs without a separate CSV."""
    data = {
        "title": [
            "The Shawshank Redemption", "The Godfather", "The Dark Knight",
            "Pulp Fiction", "Forrest Gump", "Inception", "Interstellar",
            "Parasite", "The Matrix", "Spirited Away", "Coco", "Toy Story"
        ],
        "genre": [
            "Drama", "Crime", "Action", "Crime", "Drama", "Sci-Fi",
            "Sci-Fi", "Thriller", "Sci-Fi", "Animation", "Animation", "Animation"
        ],
        "rating": [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.5, 8.7, 8.6, 8.4, 8.3],
        "votes": [2900000, 2000000, 2900000, 2200000, 2200000, 2600000,
                  2200000, 1000000, 2100000, 850000, 600000, 1100000],
        "year": [1994, 1972, 2008, 1994, 1994, 2010, 2014, 2019, 1999, 2001, 2017, 1995],
    }
    return pd.DataFrame(data)


def standardize_columns(df):
    """Map common column names to the names used by this project."""
    aliases = {
        "title": ["title", "movie_title", "movie", "name", "original_title"],
        "genre": ["genre", "genres", "category"],
        "rating": ["rating", "imdb_rating", "imdb_score", "score", "average_rating"],
        "votes": ["votes", "vote_count", "num_votes", "number_of_votes"],
        "year": ["year", "release_year", "released_year"],
    }
    original_names = {str(col).strip().lower(): col for col in df.columns}
    rename_map = {}

    for standard_name, alternatives in aliases.items():
        for alternative in alternatives:
            if alternative in original_names:
                rename_map[original_names[alternative]] = standard_name
                break

    return df.rename(columns=rename_map)


def load_movies():
    """Load movies.csv if it exists; otherwise use the built-in sample data."""
    if DATA_FILE.exists():
        print(f"Loading dataset from: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)

    print("No movies.csv file found; using the built-in sample dataset.")
    return make_sample_data()


def clean_movies(df):
    """Clean the movie dataset and validate rating values."""
    df = standardize_columns(df).copy()

    required_columns = {"title", "genre", "rating"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Please provide title, genre, and rating columns."
        )

    df["title"] = df["title"].astype("string").str.strip()
    df["genre"] = df["genre"].astype("string").str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["title", "genre", "rating"])
    df = df[df["rating"].between(0, 10)]
    df = df[(df["title"] != "") & (df["genre"] != "")]

    if df.empty:
        raise ValueError("No valid movie records found after data cleaning.")

    if "votes" in df.columns:
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    else:
        df["votes"] = pd.NA

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = pd.NA

    return df


def analyze_movies(df):
    """Print key statistics and return a summary by genre."""
    print("\\n" + "=" * 45)
    print("       MOVIE RATING ANALYSIS")
    print("=" * 45)
    print(f"Movies analyzed: {len(df)}")
    print(f"Average rating: {df['rating'].mean():.2f}/10")
    print(f"Highest rating: {df['rating'].max():.2f}/10")
    print(f"Lowest rating: {df['rating'].min():.2f}/10")

    print("\\nTOP 10 HIGHEST-RATED MOVIES")
    top_movies = df.sort_values("rating", ascending=False).head(10)
    print(top_movies[["title", "genre", "rating"]].to_string(index=False))

    genre_summary = (
        df.groupby("genre")
        .agg(
            movie_count=("title", "count"),
            average_rating=("rating", "mean")
        )
        .sort_values("average_rating", ascending=False)
        .reset_index()
    )
    genre_summary["average_rating"] = genre_summary["average_rating"].round(2)

    print("\\nAVERAGE RATING BY GENRE")
    print(genre_summary.to_string(index=False))

    return genre_summary


def save_outputs(df, genre_summary):
    """Save cleaned data, summary CSVs, and charts."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    df.to_csv(OUTPUT_DIR / "cleaned_movies.csv", index=False)
    genre_summary.to_csv(OUTPUT_DIR / "genre_summary.csv", index=False)

    sns.set_theme()

    top_movies = df.nlargest(min(10, len(df)), "rating").sort_values("rating")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_movies, x="rating", y="title")
    plt.title("Top-Rated Movies")
    plt.xlabel("Rating (out of 10)")
    plt.ylabel("Movie")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_rated_movies.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=genre_summary, x="average_rating", y="genre")
    plt.title("Average Movie Rating by Genre")
    plt.xlabel("Average Rating (out of 10)")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_rating_by_genre.png", dpi=150)
    plt.close()

    print(f"\\nFiles saved successfully in: {OUTPUT_DIR}")
    print("- cleaned_movies.csv")
    print("- genre_summary.csv")
    print("- top_rated_movies.png")
    print("- average_rating_by_genre.png")


def main():
    movies = load_movies()
    movies = clean_movies(movies)
    genre_summary = analyze_movies(movies)
    save_outputs(movies, genre_summary)


if __name__ == "__main__":
    main()
