-- 1) Build a comma-quoted list of all distinct clusters
DECLARE clusters STRING;
SET clusters = (
  SELECT
    STRING_AGG(FORMAT('"%s"', cluster))
  FROM (
    SELECT DISTINCT cluster
    FROM `issachar-feature-library.jlk.cluster_portfolio_returns`
  )
);

-- 2) Pivot + join factor and index returns with proper date conversions
EXECUTE IMMEDIATE FORMAT("""
WITH
  -- shift the original TIMESTAMP forward one day, then cast to DATE
  cluster_returns AS (
    SELECT
      DATE_ADD(CAST(DATE(date) AS DATE), INTERVAL 1 DAY) AS date,
      cluster,
      total_return
    FROM `issachar-feature-library.jlk.cluster_portfolio_returns`
  ),

  -- pivot so each cluster becomes its own column
  pivoted AS (
    SELECT *
    FROM cluster_returns
    PIVOT (
      SUM(total_return) FOR cluster IN (%s)
    )
  ),

  -- convert your Index_Returns DATETIME → DATE
  index_conv AS (
    SELECT
      DATE(date) AS date,
      * EXCEPT(date)
    FROM `josh-risk.IssacharReporting.Index_Returns`
  ),

  -- convert your factor_yields STRING → DATE using the 'YYYY-MM-DD' pattern
  factor_conv AS (
    SELECT
      PARSE_DATE('%%Y-%%m-%%d', date) AS date,
      * EXCEPT(date)
    FROM `issachar-feature-library.core_raw.factor_yields`
  )

SELECT
  p.*,
  ic.*,
  fc.*
FROM pivoted    AS p
LEFT JOIN index_conv  AS ic USING(date)
LEFT JOIN factor_conv AS fc USING(date)
ORDER BY p.date;
""", clusters);
