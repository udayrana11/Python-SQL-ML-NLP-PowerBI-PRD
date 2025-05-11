# Amazon Reviews Data Analytics

## Overview

This project demonstrates a comprehensive analysis of Amazon review data using Python. The analysis spans data extraction, cleaning, aggregation, sentiment analysis, and visualization. By leveraging libraries such as pandas, NumPy, Matplotlib, Seaborn, and TextBlob, the project uncovers insights related to user review behavior, product popularity, and sentiment distributions. The goal is to generate actionable insights that can guide business decisions and illustrate expert-level data analytics capabilities.

## Data Sources

- **CSV File & SQLite Database:**  
  The project starts by loading review data from a CSV and then connects to a SQLite database. Reviews are extracted from the `REVIEWS` table using SQL queries. This dual approach ensures robust data handling:
  - **CSV file:** Initial exploration and data ingestion.
  - **SQLite database:** Querying and further extraction of the `REVIEWS` data.

## Project Structure


- **data/**: Contains sample data (both CSV and SQLite formats).  
- **notebooks/**: Jupyter Notebook outlining the full analysis workflow.  
- **README.md**: This file, providing an overview and context.  
- **requirements.txt**: List of required Python packages (e.g., pandas, numpy, matplotlib, seaborn, textblob, wordcloud).

## Analysis Workflow

### 1. Data Extraction & Cleaning

- **Data Loading:**  
  The script loads data both via `pd.read_csv` and using a SQLite connection with `pd.read_sql_query`.  
- **Validity Check for Helpfulness:**  
  Since the review's helpfulness score must be logical, reviews where `HelpfulnessNumerator` exceeds `HelpfulnessDenominator` are filtered out, ensuring data quality.
- **Deduplication:**  
  Duplicate entries are removed based on key fields (`UserId`, `ProfileName`, `Time`, `Text`).
- **Time Conversion:**  
  Unix timestamps in the `Time` field are converted into human-readable datetime objects to facilitate time-based analysis.

### 2. User Engagement & Product Analysis

- **User Aggregation:**  
  Reviews are aggregated by `UserId` to compute:
  - The total count of summaries and texts (indicating review frequency).
  - Average review `Score`.
  - Count of products reviewed (an indicator of purchasing behavior).  
  This aggregated data is then visualized using a bar plot to highlight users who review many products.
  
- **Product Frequency:**  
  Product counts are computed to identify frequently reviewed products (with more than 500 reviews). The analysis then filters the dataset to focus on these popular products and uses Seaborn to visualize the distribution of review scores per product.

### 3. Viewer Type Classification & Text Analysis

- **Viewer Classification:**  
  Users are segmented into "Frequent" and "Not Frequent" based on whether their review count exceeds a threshold (e.g., > 50 reviews).
- **Word Count Calculation:**  
  The script calculates the number of words in each review’s `Text` to compare review lengths. Box plots are created to visualize the distribution of word counts across different viewer types. This helps in understanding if frequent reviewers write denser or longer reviews.

### 4. Sentiment Analysis

- **TextBlob Integration:**  
  The sentiment of review summaries is examined using TextBlob. For each summary, polarity (ranging from -1 to 1) and subjectivity scores are computed.
- **Sentiment Distribution:**  
  Histograms are generated to show the distribution of sentiment polarity across all reviews. This visualization provides a quick overview of how positive or negative the general sentiment is in the dataset.

### 5. Frequent Word Analysis & Word Clouds

- **Word Frequency Counting:**  
  Reviews are categorized into negative (polarity < 0) and positive (polarity > 0) groups. For each set, text is tokenized (split by spaces), and common words are tallied.
- **Word Cloud Visualization:**  
  WordCloud libraries are used to generate visualizations that highlight the most common words in negative and positive reviews respectively. These visuals aid in quickly grasping the focus topics in customer sentiments.

## Key Outcomes & Insights

- **User Reviews & Behavior:**  
  The aggregation by `UserId` identifies top reviewers and reveals that a small group of frequent reviewers contribute significantly to the review counts.
  
- **Product Engagement:**  
  Specific products with high review counts (over 500) spotlight items with greater customer engagement, which can inform marketing and inventory decisions.
  
- **Sentiment Trends:**  
  Sentiment analysis indicates a varied polarity distribution, showing a balanced mix of positive and negative feedback. The word clouds further highlight common themes in customer opinions.
  
- **Viewer Engagement:**  
  Comparative analysis of word count distributions suggests similar textual behavior between frequent and less frequent reviewers, indicating that review length may not differ dramatically by reviewer type.

## Requirements

pip install pandas numpy matplotlib seaborn sqlite3 textblob wordcloud

## Database
1. https://drive.google.com/file/d/1a4FhKPAo6rilwS3HxwBr3MFX4gkjhP1T/view?usp=drive_link
2. https://drive.google.com/file/d/1vIY83t40v9FDN4OA9nJx0PbO44aikd8o/view?usp=sharing

---

## 📝 Author

**Udaybhan Singh Rana**  
🔗 [LinkedIn](https://www.linkedin.com/in/udaybhan-rana/)

---