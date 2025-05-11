/*
   Art Gallery Analysis SQL Report
   =================================

   Overview:
   This SQL script is a comprehensive analysis report for an art gallery database, which
   tracks artists, artworks, museums, and associated pricing details. The queries in this
   report analyze various aspects such as painting distribution, pricing anomalies, museum
   operating hours, and style trends. The goal is to generate actionable insights
   valuable for curators, management, and technical recruiters.

   Database Tables Previewed:
     - artist: Detailed information about the artists.
     - canvas_size: Dimensions and descriptive labels for canvases.
     - image_link: Links to images of the paintings.
     - museum: Museum details including names, locations, and countries.
     - museum_hours: Operating hours for each museum, by day.
     - product_size: Pricing and product size details.
     - subject: Classification of paintings by subject.
     - work: Records of paintings along with relevant links to artists and museums.

   Author: UBS
   Date: 2025-05-08
   Repository: (Insert your GitHub repository URL here)

   Notes:
     - Each section is carefully commented to provide clarity regarding the intention,
       methodology, and any assumptions.
     - The script features advanced SQL constructs like CTEs, window functions, and conditional
       ordering to maintain efficiency and readability.
     - This file is designed to demonstrate expert SQL skills and is ideal for technical reviews,
       portfolio submissions, and recruiters assessing your technical prowess.
*/

/* -----------------------------------------------------
   Preview Relevant Tables to Understand Structure and Data
   ----------------------------------------------------- */
SELECT * FROM artist;
SELECT * FROM canvas_size;
SELECT * FROM image_link;
SELECT * FROM museum;
SELECT * FROM museum_hours;
SELECT * FROM product_size;
SELECT * FROM subject;
SELECT * FROM work;


/* -----------------------------------------------------
   1. Count of Paintings Not Displayed in Any Museum
   -----------------------------------------------------
   Fetches the total count of paintings that are not associated
   with any museum (i.e., museum_id is NULL in the work table).
*/
SELECT COUNT(work_id) AS paintings_not_in_museum
FROM work
WHERE museum_id IS NULL;


/* -----------------------------------------------------
   2. Count of Overpriced Paintings
   -----------------------------------------------------
   Counts the number of unique paintings where the sale price is
   greater than the regular price, indicating an overpriced scenario.
*/
SELECT COUNT(DISTINCT work_id) AS overpriced_paintings
FROM product_size
WHERE sale_price > regular_price;


/* -----------------------------------------------------
   3. Count of Heavily Discounted Paintings
   -----------------------------------------------------
   Identifies the unique paintings whose sale price is less than
   50% of the regular price, showcasing heavy discounts.
*/
SELECT COUNT(DISTINCT work_id) AS heavily_discounted_paintings
FROM product_size
WHERE sale_price < regular_price / 2;


/* -----------------------------------------------------
   4. Canvas Size With the Highest Sale Price
   -----------------------------------------------------
   Leverages a Common Table Expression (CTE) to join canvas dimensions with sale
   prices. It filters for numeric keys and then groups by dimensions to find the highest sale price.
*/
WITH canvas_pricing AS (
    SELECT 
        c.size_id,
        c.width,
        c.height,
        p.sale_price
    FROM canvas_size c
    JOIN product_size p ON c.size_id = p.size_id::bigint
    WHERE p.size_id ~ '^\d+$'
)
SELECT 
    width, 
    height, 
    MAX(sale_price) AS highest_sale_price
FROM canvas_pricing
GROUP BY width, height
ORDER BY highest_sale_price DESC
LIMIT 1;


/* -----------------------------------------------------
   5. Top 10 Most Expensive Paintings (Subject-wise)
   -----------------------------------------------------
   Retrieves the painting names for the top 10 most expensive paintings by
   using a CTE to join work and product_size, sorted by sale_price.
*/
WITH price_data AS (
    SELECT 
        w.name AS painting_name,
        p.sale_price
    FROM product_size p
    JOIN work w ON p.work_id = w.work_id
)
SELECT painting_name
FROM price_data
ORDER BY sale_price DESC
LIMIT 10;


/* -----------------------------------------------------
   6. Museums Open on Both Sunday and Monday
   -----------------------------------------------------
   Identifies museums that operate on both Sunday and Monday by joining museum and museum_hours,
   grouping by museum name and city, and ensuring two distinct days are present.
*/
WITH open_days AS (
    SELECT 
        m.name, 
        m.city, 
        mh.day
    FROM museum m
    JOIN museum_hours mh ON m.museum_id = mh.museum_id
)
SELECT name, city
FROM open_days
WHERE day IN ('Sunday', 'Monday')
GROUP BY name, city
HAVING COUNT(DISTINCT day) = 2;


/* -----------------------------------------------------
   7. Count of Museums Open All 7 Days of the Week
   -----------------------------------------------------
   Determines the number of museums that are operational every day by
   grouping museum_hours by museum_id and counting distinct days.
*/
SELECT COUNT(*) AS always_open_museums
FROM (
    SELECT museum_id
    FROM museum_hours
    GROUP BY museum_id
    HAVING COUNT(DISTINCT day) = 7
) AS sub;


/* -----------------------------------------------------
   8. Most and Least Popular Painting Styles Based on Sale Price
   -----------------------------------------------------
   Analyzes painting styles to extract the top 3 styles (by highest maximum sale price)
   and the bottom 3 (by lowest minimum sale price). Liquids the results using UNION ALL.
*/
WITH style_sales AS (
    SELECT 
        w.style,
        ps.sale_price
    FROM work w
    JOIN product_size ps ON w.work_id = ps.work_id
),
most_popular AS (
    SELECT style 
    FROM style_sales
    GROUP BY style
    ORDER BY MAX(sale_price) DESC
    LIMIT 3
),
least_popular AS (
    SELECT style 
    FROM style_sales
    GROUP BY style
    ORDER BY MIN(sale_price) ASC
    LIMIT 3
)
SELECT 'Most Popular' AS category, style FROM most_popular
UNION ALL
SELECT 'Least Popular', style FROM least_popular;


/* -----------------------------------------------------
   9. Artists With Paintings Displayed in Multiple Countries
   -----------------------------------------------------
   Identifies artists whose paintings are exhibited in museums across
   multiple countries by grouping based on artist_id and counting distinct countries.
*/
SELECT w.artist_id
FROM work w
JOIN museum m ON w.museum_id = m.museum_id
GROUP BY w.artist_id
HAVING COUNT(DISTINCT m.country) > 1;


/* -----------------------------------------------------
   10. City and Country With the Highest Number of Museums
   -----------------------------------------------------
   Aggregates museum counts per country and city, then extracts the locations with the highest counts.
   Note: The query uses string aggregation to list all locations with matching max counts.
*/
WITH country_counts AS (
    SELECT country, COUNT(*) AS total FROM museum GROUP BY country
),
city_counts AS (
    SELECT city, COUNT(*) AS total FROM museum GROUP BY city
),
max_country AS (
    SELECT MAX(total) AS max_count FROM country_counts
),
max_city AS (
    SELECT MAX(total) AS max_count FROM city_counts
)
SELECT 
    STRING_AGG(cc.country, ', ') AS top_countries,
    STRING_AGG(ci.city, ', ') AS top_cities
FROM country_counts cc
JOIN max_country mc ON cc.total = mc.max_count
JOIN city_counts ci
JOIN max_city mci ON ci.total = mci.max_count;


/* -----------------------------------------------------
   11. Most and Least Expensive Paintings and Their Full Context
   -----------------------------------------------------
   Provides comprehensive details (artist, museum, canvas size, etc.) for the most expensive and
   least expensive paintings by using a CTE to gather all related information.
*/
WITH painting_data AS (
    SELECT 
        a.full_name,
        w.name AS painting_name,
        ps.sale_price,
        m.name AS museum_name,
        m.city,
        cs.label
    FROM work w
    JOIN artist a ON w.artist_id = a.artist_id
    JOIN museum m ON w.museum_id = m.museum_id
    JOIN product_size ps ON w.work_id = ps.work_id
    JOIN canvas_size cs ON cs.size_id = ps.size_id::bigint
    WHERE ps.size_id ~ '^\d+$'
),
price_extremes AS (
    SELECT MAX(sale_price) AS max_price, MIN(sale_price) AS min_price FROM painting_data
)
SELECT *
FROM painting_data
WHERE sale_price = (SELECT max_price FROM price_extremes)
   OR sale_price = (SELECT min_price FROM price_extremes);


/* -----------------------------------------------------
   12. Top 5 Most Popular Museums by Number of Paintings
   -----------------------------------------------------
   Counts the total paintings for museums (where museum_id exists) and returns the top five institutions.
*/
SELECT museum_id, COUNT(*) AS total_paintings
FROM work
WHERE museum_id IS NOT NULL
GROUP BY museum_id
ORDER BY total_paintings DESC
LIMIT 5;


/* -----------------------------------------------------
   13. Top 5 Most Popular Artists by Number of Paintings
   -----------------------------------------------------
   Aggregates painting counts per artist and sorts to determine the five most prolific artists.
*/
SELECT artist_id, COUNT(*) AS total_paintings
FROM work
GROUP BY artist_id
ORDER BY total_paintings DESC
LIMIT 5;


/* -----------------------------------------------------
   14. The 3 Least Popular Canvas Sizes Based on Sale Price
   -----------------------------------------------------
   Displays the three canvas sizes associated with the lowest sale prices,
   indicating less market popularity.
*/
WITH canvas_sales AS (
    SELECT 
        c.width, 
        c.height, 
        c.label, 
        p.sale_price
    FROM canvas_size c
    JOIN product_size p ON c.size_id = p.size_id::bigint
    WHERE p.size_id ~ '^\d+$'
)
SELECT width, height, label
FROM canvas_sales
ORDER BY sale_price ASC
LIMIT 3;


/* -----------------------------------------------------
   15. Museum Open the Longest in a Day
   -----------------------------------------------------
   Converts the stored opening and closing times (after removing any extra spaces)
   into comparable time formats, then calculates the open duration for each day to select the museum with the longest hours.
*/
WITH open_hours AS (
    SELECT 
        m.name, 
        m.state, 
        mh.day,
        TO_TIMESTAMP(REPLACE(open, ' ', ''), 'HH12:MI:PM')::time AS open_time,
        TO_TIMESTAMP(REPLACE(close, ' ', ''), 'HH12:MI:PM')::time AS close_time
    FROM museum m
    JOIN museum_hours mh ON m.museum_id = mh.museum_id
)
SELECT 
    name, 
    state, 
    day, 
    (close_time - open_time) AS open_duration
FROM open_hours
ORDER BY open_duration DESC
LIMIT 1;


/* -----------------------------------------------------
   16. Museum with Most High-Priced Paintings of the Same Style
   -----------------------------------------------------
   Retrieves the museum and corresponding style records resulting in high sale prices.
   Although further grouping by style and museum might be considered, this query orders by sale price to highlight high-value trends.
*/
WITH style_prices AS (
    SELECT 
        w.museum_id, 
        w.style, 
        p.sale_price
    FROM work w
    JOIN product_size p ON w.work_id = p.work_id
    WHERE w.museum_id IS NOT NULL
)
SELECT museum_id, style
FROM style_prices
ORDER BY sale_price DESC
LIMIT 5;


/* -----------------------------------------------------
   17. Country with the 5th Highest Number of Paintings
   -----------------------------------------------------
   Uses a window function (ROW_NUMBER) to rank countries based on their total painting count,
   and then selects the country ranked 5th.
*/
WITH ranked_nationality AS (
    SELECT 
        a.nationality, 
        COUNT(w.work_id) AS painting_count,
        ROW_NUMBER() OVER (ORDER BY COUNT(w.work_id) DESC) AS rn
    FROM artist a
    JOIN work w ON a.artist_id = w.artist_id
    GROUP BY a.nationality
)
SELECT nationality, painting_count
FROM ranked_nationality
WHERE rn = 5;


/* -----------------------------------------------------
   18. Artist With the Most Portrait Paintings Outside the USA
   -----------------------------------------------------
   Analyzes portrait paintings exhibited in museums outside the USA. The query aggregates by artist
   and selects the one with the maximum count.
*/
WITH non_us_portraits AS (
    SELECT 
        a.full_name, 
        a.nationality, 
        COUNT(w.work_id) AS total_paintings
    FROM artist a
    JOIN work w ON a.artist_id = w.artist_id
    JOIN museum m ON w.museum_id = m.museum_id
    WHERE w.subject = 'Portrait' AND m.country != 'USA'
    GROUP BY a.full_name, a.nationality
    ORDER BY total_paintings DESC
    LIMIT 1
)
SELECT * FROM non_us_portraits;
