# Movie Recommendation System

This project is a **Content-Based Movie Recommendation System** built using Python. It suggests movies similar to a user-provided movie title by analyzing textual features like genres, cast, director, keywords, and tagline using TF-IDF vectorization and cosine similarity.

## Features

- Recommends top 10 movies similar to a given movie title.
- Uses **TF-IDF** and **Cosine Similarity** to compute closeness.
- Automatically handles missing data and matches close titles using `difflib`.
- Simple and intuitive CLI interface for users.


# Content-Based Movie Recommendation System

This project is a **content-based movie recommender system** built using Python. It intelligently suggests movies similar to a user's input by analyzing features such as genre, cast, director, and keywords. Using **TF-IDF vectorization** and **cosine similarity**, it matches content patterns to provide personalized recommendations.

---

## Dataset Overview

The dataset contains **3,746 movies** with rich metadata. Here's a snapshot of the schema:

| Column                | Type            | Description                                      |
|-----------------------|-----------------|--------------------------------------------------|
| `budget`              | int64           | Budget in USD                                    |
| `genres`              | object          | List of genres                                   |
| `id`                  | int64           | Movie ID                                         |
| `keywords`            | object          | Keywords describing plot/themes                  |
| `original_language`   | object          | Original language of the film                    |
| `original_title`      | object          | Original title                                   |
| `overview`            | object          | Brief synopsis                                   |
| `popularity`          | float64         | Popularity score                                 |
| `production_companies`| object          | List of production houses                        |
| `production_countries`| object          | Country/countries of production                  |
| `release_date`        | datetime64[ns]  | Release date                                     |
| `revenue`             | int64           | Revenue in USD                                   |
| `runtime`             | float64         | Duration in minutes                              |
| `spoken_languages`    | object          | Languages spoken                                 |
| `status`              | object          | Release status (e.g., Released)                  |
| `tagline`             | object          | Movie tagline                                    |
| `title`               | object          | Movie title                                      |
| `vote_average`        | float64         | Average user rating                              |
| `vote_count`          | int64           | Number of votes received                         |
| `cast`                | object          | List of movie cast members                        |
| `crew`                | object          | List of movie crew members                        |
| `director`            | object          | Movie director                                   |


---

## Libraries & Tools

- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning models, including TF-IDF vectorization and similarity computation.
- **Difflib**: For finding the closest matching movie titles.
- **Matplotlib & Seaborn**: For visualizations (if required).
- **Imbalanced-learn (SMOTE)**: For handling class imbalances (optional, depending on dataset use).

---


## Result

🎥 Enter a movie name you like: Inception
✅ Closest match found: *Inception*

🍿 Movies recommended for you:

1. 🎬 Interstellar  
2. 🎬 The Dark Knight  
3. 🎬 The Prestige  
4. 🎬 Memento  
5. 🎬 Tenet  
6. 🎬 Shutter Island  
7. 🎬 The Matrix  
8. 🎬 In Time  
9. 🎬 Source Code  
10. 🎬 Limitless

✅ Enjoy your watchlist!

---

## 📝 Author

**Udaybhan Singh Rana**  
🔗 [LinkedIn](https://www.linkedin.com/in/udaybhan-rana/)

---


