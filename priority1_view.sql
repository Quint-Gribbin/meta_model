-- VARIABLES FIRST
DECLARE clusters    STRING;
DECLARE pivot_cols  STRING;
DECLARE factor_cols STRING;

-- 1) clusters for the PIVOT
SET clusters = (
  SELECT STRING_AGG(FORMAT('"%s"', cluster))
  FROM  (SELECT DISTINCT cluster
         FROM `issachar-feature-library.jlk.cluster_portfolio_returns`)
);

-- 2) COALESCE list for pivoted clusters
SET pivot_cols = (
  SELECT STRING_AGG(
           FORMAT('COALESCE(p.`%s`,0) AS `%s`', cluster, cluster),
           ', ')
  FROM  (SELECT DISTINCT cluster
         FROM `issachar-feature-library.jlk.cluster_portfolio_returns`)
);

-- 3) COALESCE list for factor_yields
SET factor_cols = (
  SELECT STRING_AGG(
           FORMAT('COALESCE(`%s`,0) AS `%s`', column_name, column_name),
           ', ' ORDER BY ordinal_position)
  FROM  `issachar-feature-library.core_raw.INFORMATION_SCHEMA.COLUMNS`
  WHERE table_name = 'factor_yields'
    AND column_name <> 'date'
);

-- MAIN QUERY
EXECUTE IMMEDIATE FORMAT("""
WITH
  -- shift returns down one trading row (no calendar-day add)
  cluster_returns AS (
    SELECT
      CAST(date AS DATE) AS date,
      cluster,
      LAG(total_return) OVER (PARTITION BY cluster ORDER BY date) AS total_return
    FROM `issachar-feature-library.jlk.cluster_portfolio_returns`
  ),

  -- pivot clusters
  pivoted AS (
    SELECT * FROM cluster_returns
    PIVOT (SUM(total_return) FOR cluster IN (%s))
  ),

  -- NULL→0 for every cluster column
  pivoted_clean AS (
    SELECT
      p.date,
      %s
    FROM pivoted AS p
  ),

  -- index returns (leave as-is)
  index_conv AS (
    SELECT DATE(date) AS date, * EXCEPT(date)
    FROM   `josh-risk.IssacharReporting.Index_Returns`
  ),

  -- factor_yields NULL→0
  factor_conv AS (
    SELECT
      PARSE_DATE('%%Y-%%m-%%d', date) AS date,
      %s
    FROM `issachar-feature-library.core_raw.factor_yields`
  )

SELECT
  pc.date,                 -- one date column only
  pc.* EXCEPT(date),       -- cluster columns
  ic.* EXCEPT(date),       -- index-return columns
  fc.* EXCEPT(date)        -- factor-yield columns
FROM   pivoted_clean AS pc
LEFT   JOIN index_conv  AS ic USING(date)
LEFT   JOIN factor_conv AS fc USING(date)
ORDER  BY pc.date;
""",
  clusters,
  pivot_cols,
  factor_cols);
