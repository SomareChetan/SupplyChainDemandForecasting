-- =====================================================
-- SUPPLY CHAIN DEMAND FORECASTING PROJECT
-- Person A - Business Analytics Queries
-- =====================================================


-- =====================================================
-- 1. Total Sales
-- =====================================================

SELECT SUM(Weekly_Sales) AS Total_Sales
FROM walmart_sales;


-- =====================================================
-- 2. Top 10 Stores by Sales
-- =====================================================

SELECT
    Store,
    SUM(Weekly_Sales) AS Total_Sales
FROM walmart_sales
GROUP BY Store
ORDER BY Total_Sales DESC
LIMIT 10;


-- =====================================================
-- 3. Sales by Department
-- =====================================================

SELECT
    Dept,
    SUM(Weekly_Sales) AS Total_Sales
FROM walmart_sales
GROUP BY Dept
ORDER BY Total_Sales DESC;


-- =====================================================
-- 4. Monthly Sales Trend
-- =====================================================

SELECT
    Month_Name,
    SUM(Weekly_Sales) AS Total_Sales
FROM walmart_sales
GROUP BY Month_Name
ORDER BY Month;


-- =====================================================
-- 5. Quarterly Sales
-- =====================================================

SELECT
    Quarter,
    SUM(Weekly_Sales) AS Total_Sales
FROM walmart_sales
GROUP BY Quarter
ORDER BY Quarter;


-- =====================================================
-- 6. Holiday vs Non-Holiday Sales
-- =====================================================

SELECT
    IsHoliday,
    SUM(Weekly_Sales) AS Total_Sales
FROM walmart_sales
GROUP BY IsHoliday;


-- =====================================================
-- 7. Sales by Store Type
-- =====================================================

SELECT
    Type,
    AVG(Weekly_Sales) AS Average_Sales
FROM walmart_sales
GROUP BY Type;


-- =====================================================
-- 8. Highest Selling Week
-- =====================================================

SELECT
    Week,
    SUM(Weekly_Sales) AS Total_Sales
FROM walmart_sales
GROUP BY Week
ORDER BY Total_Sales DESC
LIMIT 10;


-- =====================================================
-- 9. Top Departments
-- =====================================================

SELECT
    Dept,
    AVG(Weekly_Sales) AS Average_Sales
FROM walmart_sales
GROUP BY Dept
ORDER BY Average_Sales DESC
LIMIT 10;


-- =====================================================
-- 10. Store Performance
-- =====================================================

SELECT
    Store,
    COUNT(*) AS Transactions,
    AVG(Weekly_Sales) AS Average_Sales
FROM walmart_sales
GROUP BY Store
ORDER BY Average_Sales DESC;