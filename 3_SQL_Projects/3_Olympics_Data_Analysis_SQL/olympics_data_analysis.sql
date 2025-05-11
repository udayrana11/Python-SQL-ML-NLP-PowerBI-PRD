-- ============================================================
-- Olympics Data Analysis SQL Queries
-- Author: UBS
-- Purpose: Extract meaningful insights from historical Olympic data.
-- Tables used:
--   olympics_history
--   olympics_history_noc_regions
-- ============================================================

/* ===================================================================
   Basic Data Extraction
   =================================================================== */

-- Display all records from the "olympics_history_noc_regions" table.
-- This query fetches all data for inspection. Typically, avoid SELECT * in production.
SELECT noc, region, country FROM olympics_history_noc_regions;

-- How many Olympic Games have been held?
-- This query counts the distinct Olympic Games held in history.
SELECT COUNT(DISTINCT games) AS total_olympic_games
FROM olympics_history;

-- List all Olympic Games held so far.
-- Retrieves a list of all distinct Olympic Games that have occurred.
SELECT DISTINCT games FROM olympics_history;

-- Total number of nations that participated in each Olympic game.
-- This query calculates the number of unique countries participating in the Olympics.
SELECT COUNT(DISTINCT region) AS total_nations
FROM olympics_history_noc_regions;

/* ===================================================================
   Country Participation Analysis
   =================================================================== */

-- Which year saw the highest and lowest number of countries participating?
-- This identifies the years with the most and least country participation.
SELECT year, COUNT(DISTINCT region) AS number_of_countries
FROM olympics_history oh
JOIN olympics_history_noc_regions ohn ON oh.noc = ohn.noc
GROUP BY year
ORDER BY number_of_countries DESC;

-- Count distinct regions per year and rank by participation.
-- Displays the number of countries that participated in each year.
SELECT year, COUNT(DISTINCT region) AS number_of_countries
FROM olympics_history oh
JOIN olympics_history_noc_regions ohn ON oh.noc = ohn.noc
GROUP BY year
ORDER BY number_of_countries DESC;

/* ===================================================================
   Continuous Participation & Sport Analysis
   =================================================================== */

-- Which nation has participated in all of the Olympic Games?
-- This query identifies nations that have participated in every Olympic Games.
WITH total_olympic_years AS (
  SELECT COUNT(DISTINCT year) AS total_years
  FROM olympics_history
),
region_participation AS (
  SELECT region, COUNT(DISTINCT year) AS participated_years
  FROM olympics_history oh
  JOIN olympics_history_noc_regions nr ON oh.noc = nr.noc
  GROUP BY region
)
SELECT rp.region
FROM region_participation rp
JOIN total_olympic_years toy ON rp.participated_years = toy.total_years;

-- Identify the sport which was played in all Summer Olympics.
-- This identifies the sport that appeared in every Summer Olympics.
WITH summer_years AS (
  SELECT DISTINCT year
  FROM olympics_history
  WHERE season = 'Summer'
),
sport_year_counts AS (
  SELECT sport, COUNT(DISTINCT year) AS cnt
  FROM olympics_history
  WHERE season = 'Summer'
  GROUP BY sport
),
total_summer_years AS (
  SELECT COUNT(*) AS total_years
  FROM summer_years
)
SELECT s.sport
FROM sport_year_counts s, total_summer_years t
WHERE s.cnt = t.total_years;

-- Which sports were played only once in the Olympics?
-- This identifies sports that appeared in only one Olympic game.
SELECT sport
FROM olympics_history
GROUP BY sport
HAVING COUNT(DISTINCT year) = 1;

/* ===================================================================
   Sports & Athlete Analysis
   =================================================================== */

-- Fetch the total number of sports played in each Olympic game.
-- This calculates how many different sports were played in each Olympic event.
SELECT games, COUNT(DISTINCT sport) AS sport_played
FROM olympics_history
GROUP BY games;

-- Fetch details of the oldest athletes to win a gold medal.
-- This fetches the names, ages, and other details of the oldest athletes who won a gold medal.
SELECT name, age, medal, sex, height, weight, team
FROM olympics_history
WHERE medal = 'Gold' AND age IS NOT NULL
ORDER BY age DESC;

-- Find the ratio of male and female athletes that participated in all Olympic games.
-- This calculates the gender distribution of Olympic participants.
WITH gender_counts AS (
  SELECT
    COUNT(CASE WHEN sex = 'M' THEN 1 END) AS male,
    COUNT(CASE WHEN sex = 'F' THEN 1 END) AS female,
    COUNT(*) AS total
  FROM olympics_history
)
SELECT
  male,
  female,
  total,
  ROUND(male * 100.0 / total, 2) AS male_percentage,
  ROUND(female * 100.0 / total, 2) AS female_percentage
FROM gender_counts;

-- Fetch the top 5 athletes who have won the most gold medals.
-- This returns the top 5 athletes with the highest gold medal count.
SELECT name, COUNT(medal) AS gold_medals
FROM olympics_history
WHERE medal = 'Gold'
GROUP BY name
ORDER BY COUNT(medal) DESC
LIMIT 5;

-- Fetch the top 5 athletes who have won the most medals (gold/silver/bronze in total).
-- This lists the top 5 athletes with the most overall medals.
SELECT name, COUNT(medal) AS total_medals
FROM olympics_history
GROUP BY name
ORDER BY COUNT(medal) DESC
LIMIT 5;

/* ===================================================================
   Country & Medal Distribution Analysis
   =================================================================== */

-- Ranking of countries by total number of medals won.
-- This identifies which countries have won the most Olympic medals.
SELECT noc, COUNT(medal) AS total_medals
FROM olympics_history
GROUP BY noc
ORDER BY COUNT(medal) DESC;

-- List total gold, silver, and bronze medals won by each country.
-- This query calculates the number of gold, silver, and bronze medals won by each country.
SELECT noc, 
       SUM(CASE WHEN medal = 'Gold' THEN 1 ELSE 0 END) AS total_gold,
       SUM(CASE WHEN medal = 'Silver' THEN 1 ELSE 0 END) AS total_silver,
       SUM(CASE WHEN medal = 'Bronze' THEN 1 ELSE 0 END) AS total_bronze
FROM olympics_history
GROUP BY noc;

-- List medals by type won by each country for each Olympic game.
-- This details the number of gold, silver, and bronze medals won by each country per Olympic game.
SELECT games, noc, 
       SUM(CASE WHEN medal = 'Gold' THEN 1 ELSE 0 END) AS total_gold,
       SUM(CASE WHEN medal = 'Silver' THEN 1 ELSE 0 END) AS total_silver,
       SUM(CASE WHEN medal = 'Bronze' THEN 1 ELSE 0 END) AS total_bronze
FROM olympics_history
GROUP BY games, noc;

-- Identify which country won the most gold, silver, and bronze medals in each Olympic game.
-- This shows the countries with the highest medal counts in each Olympics.
SELECT games, noc, 
       SUM(CASE WHEN medal = 'Gold' THEN 1 ELSE 0 END) AS total_gold,
       SUM(CASE WHEN medal = 'Silver' THEN 1 ELSE 0 END) AS total_silver,
       SUM(CASE WHEN medal = 'Bronze' THEN 1 ELSE 0 END) AS total_bronze,
       COUNT(medal) AS total_medals
FROM olympics_history
GROUP BY games, noc
ORDER BY total_medals DESC
LIMIT 10;

-- Which countries have never won a gold medal but have won silver/bronze medals?
-- This identifies countries with silver/bronze medals but no gold.
SELECT noc, 
       SUM(CASE WHEN medal = 'Gold' THEN 1 ELSE 0 END) AS total_gold,
       SUM(CASE WHEN medal = 'Silver' THEN 1 ELSE 0 END) AS total_silver,
       SUM(CASE WHEN medal = 'Bronze' THEN 1 ELSE 0 END) AS total_bronze,
       COUNT(medal) AS total_medals
FROM olympics_history
GROUP BY noc
HAVING SUM(CASE WHEN medal = 'Gold' THEN 1 ELSE 0 END) = 0
   AND SUM(CASE WHEN medal = 'Silver' THEN 1 ELSE 0 END) > 0
   AND SUM(CASE WHEN medal = 'Bronze' THEN 1 ELSE 0 END) > 0
ORDER BY total_medals DESC;

/* ===================================================================
   India-Specific Analysis
   =================================================================== */

-- In which sport/event has India won the highest number of medals?
-- This shows the sport/event where India has the most medals.
SELECT noc, sport, event, COUNT(medal) AS medal_count
FROM olympics_history
WHERE noc = 'IND'
GROUP BY noc, sport, event
ORDER BY COUNT(medal) DESC;

-- Break down of each Olympic game where India won a medal in Hockey.
-- Lists the specific Olympic Games in which India won medals in Hockey.
SELECT games, noc, sport, event, COUNT(medal) AS medal_count
FROM olympics_history
WHERE noc = 'IND' AND sport = 'Hockey'
GROUP BY games, noc, sport, event
ORDER BY COUNT(medal) DESC;
