-- Task 1.1: Verify dataset import

SELECT COUNT(*) AS total_rows
FROM traffic;

SELECT *
FROM traffic
LIMIT 10;

-- Task 1.2: Total yearly traffic volume for 2012-2017

SELECT
    strftime('%Y', date_time) AS year,
    SUM(traffic_volume) AS total_traffic_volume
FROM traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- Task 1.2: Year-on-year percentage change and trend

WITH yearly_traffic AS (
    SELECT
        strftime('%Y', date_time) AS year,
        SUM(traffic_volume) AS total_traffic_volume
    FROM traffic
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY year
),

traffic_change AS (
    SELECT
        year,
        total_traffic_volume,
        LAG(total_traffic_volume) OVER (ORDER BY year)
            AS previous_year_traffic
    FROM yearly_traffic
)

SELECT
    year,
    total_traffic_volume,

    total_traffic_volume - previous_year_traffic
        AS change_from_previous_year,

    ROUND(
        (total_traffic_volume - previous_year_traffic)
        * 100.0 / previous_year_traffic,
        2
    ) AS percentage_change,

    CASE
        WHEN previous_year_traffic IS NULL THEN 'No comparison'
        WHEN total_traffic_volume > previous_year_traffic THEN 'Increase'
        WHEN total_traffic_volume < previous_year_traffic THEN 'Decrease'
        ELSE 'No change'
    END AS trend

FROM traffic_change
ORDER BY year;

-- Task 1.2: Check yearly data coverage

SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*) AS number_of_records,
    MIN(date_time) AS first_record,
    MAX(date_time) AS last_record
FROM traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- Task 1.2: Check unique timestamps and repeated hourly records

SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*) AS total_records,
    COUNT(DISTINCT date_time) AS unique_timestamps,
    COUNT(*) - COUNT(DISTINCT date_time) AS repeated_timestamp_records
FROM traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- Task 1.2: Inspect repeated timestamps

SELECT
    date_time,
    COUNT(*) AS rows_at_timestamp,
    COUNT(DISTINCT traffic_volume) AS distinct_traffic_values,
    MIN(traffic_volume) AS minimum_traffic_volume,
    MAX(traffic_volume) AS maximum_traffic_volume
FROM traffic
GROUP BY date_time
HAVING COUNT(*) > 1
ORDER BY rows_at_timestamp DESC, date_time
LIMIT 20;


-- Task 1.2: Check whether duplicated timestamps
-- ever contain different traffic volumes

SELECT COUNT(*) AS timestamps_with_different_traffic_values
FROM (
    SELECT date_time
    FROM traffic
    GROUP BY date_time
    HAVING COUNT(DISTINCT traffic_volume) > 1
);

-- Task 1.2: Correct yearly traffic totals
-- Count each hourly timestamp only once

WITH hourly_traffic AS (
    SELECT
        date_time,
        MAX(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
)

SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*) AS unique_hours,
    SUM(traffic_volume) AS total_traffic_volume
FROM hourly_traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- Task 1.2: Correct year-on-year change
-- using one traffic value per timestamp

WITH hourly_traffic AS (
    SELECT
        date_time,
        MAX(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
),

yearly_traffic AS (
    SELECT
        strftime('%Y', date_time) AS year,
        SUM(traffic_volume) AS total_traffic_volume
    FROM hourly_traffic
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY year
),

traffic_change AS (
    SELECT
        year,
        total_traffic_volume,
        LAG(total_traffic_volume) OVER (ORDER BY year)
            AS previous_year_traffic
    FROM yearly_traffic
)

SELECT
    year,
    total_traffic_volume,

    total_traffic_volume - previous_year_traffic
        AS change_from_previous_year,

    ROUND(
        (total_traffic_volume - previous_year_traffic)
        * 100.0 / previous_year_traffic,
        2
    ) AS percentage_change,

    CASE
        WHEN previous_year_traffic IS NULL THEN 'No comparison'
        WHEN total_traffic_volume > previous_year_traffic THEN 'Increase'
        WHEN total_traffic_volume < previous_year_traffic THEN 'Decrease'
        ELSE 'No change'
    END AS trend

FROM traffic_change
ORDER BY year;

-- Task 1.2: Key observations
--
-- Observation 1:
-- 2016 and 2017 both contain full-year data, making them the most
-- directly comparable years in the dataset. Total deduplicated traffic
-- volume increased from 25,032,183 in 2016 to 29,420,221 in 2017,
-- an increase of 4,388,038 (approximately 17.53%).
--
-- Observation 2:
-- Large year-to-year changes involving 2012, 2014 and 2015 should be
-- interpreted cautiously because these years contain incomplete data.
-- For example, 2012 only covers October to December, while 2013 covers
-- the full year. Therefore, the apparent 255.74% increase from 2012
-- to 2013 is largely influenced by differences in data coverage rather
-- than necessarily representing a real increase in traffic demand.

-- Task 1.3: Check holiday values

SELECT DISTINCT holiday
FROM traffic
ORDER BY holiday;

-- Task 1.3: Average temperature for selected holidays, 2015-2017

SELECT
    strftime('%Y', date_time) AS year,
    holiday,
    ROUND(AVG(temp), 2) AS average_temperature_kelvin
FROM traffic
WHERE strftime('%Y', date_time) IN ('2015', '2016', '2017')
  AND holiday IN ('New Years Day', 'Labor Day')
GROUP BY year, holiday
ORDER BY holiday, year;

-- Task 1.3: Compare holiday temperature and traffic conditions

WITH holiday_hourly AS (
    SELECT
        date_time,
        holiday,
        AVG(temp) AS temperature_kelvin,
        MAX(traffic_volume) AS traffic_volume
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day')
    GROUP BY date_time, holiday
)

SELECT
    strftime('%Y', date_time) AS year,
    holiday,
    ROUND(AVG(temperature_kelvin), 2) AS average_temperature_kelvin,
    ROUND(AVG(temperature_kelvin) - 273.15, 2)
        AS average_temperature_celsius,
    ROUND(AVG(traffic_volume), 2) AS average_traffic_volume
FROM holiday_hourly
WHERE strftime('%Y', date_time) IN ('2015', '2016', '2017')
GROUP BY year, holiday
ORDER BY holiday, year;


-- Task 1.3: Key observations
--
-- Labor Day:
-- Average temperature was 295.02 K (21.87°C) in 2015,
-- decreased to 293.17 K (20.02°C) in 2016,
-- then increased to 295.54 K (22.39°C) in 2017.
-- Average traffic volume was 973 in 2015, 1,064 in 2016,
-- and 1,026 in 2017.
--
-- The temperature changes were relatively small, while traffic volume
-- changed only modestly. There is no clear indication from these three
-- observations that temperature alone strongly affected Labor Day traffic.
--
-- New Years Day:
-- No 2015 result is available because the 2015 dataset begins in June.
-- Temperature increased from 265.94 K (-7.21°C) in 2016
-- to 270.62 K (-2.53°C) in 2017.
-- However, average traffic volume decreased from 1,513 in 2016
-- to 798 in 2017.
--
-- Therefore, the warmer temperature in 2017 did not correspond to
-- higher traffic volume. This suggests that temperature alone does not
-- explain the change in New Years Day traffic, and other factors may
-- also influence traffic conditions.