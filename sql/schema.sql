-- =====================================================
-- SUPPLY CHAIN DEMAND FORECASTING PROJECT
-- Person A - Database Schema
-- =====================================================

DROP TABLE IF EXISTS walmart_sales;

CREATE TABLE walmart_sales (
    Store INTEGER,
    Dept INTEGER,
    Date DATE,
    Weekly_Sales REAL,
    IsHoliday BOOLEAN,

    Temperature REAL,
    Fuel_Price REAL,

    MarkDown1 REAL,
    MarkDown2 REAL,
    MarkDown3 REAL,
    MarkDown4 REAL,
    MarkDown5 REAL,

    CPI REAL,
    Unemployment REAL,

    Type TEXT,
    Size INTEGER,

    Year INTEGER,
    Month INTEGER,
    Month_Name TEXT,
    Quarter INTEGER,
    Week INTEGER,
    Day INTEGER,
    Day_Name TEXT
);