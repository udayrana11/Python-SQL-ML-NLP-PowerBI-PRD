# 👨‍💼 Employee Salary Analysis – SQL Portfolio Project

## Overview

This project explores employee compensation data using advanced SQL techniques. The dataset simulates a real-world organizational structure across departments, and the queries demonstrate skills in **table creation**, **data insertion**, **aggregation**, **window functions**, and **ranking**.

The purpose of this analysis is to showcase strong SQL capabilities that are applicable to HR analytics, compensation benchmarking, and technical interviews.

---

## Author

**Uday Rana**  
🔗 GitHub: [@udayrana11](https://github.com/udayrana11)

---

## Schema

**Table:** `employee`

| Column     | Type        | Description                  |
|------------|-------------|------------------------------|
| emp_Id     | `INT`       | Unique employee ID           |
| emp_name   | `VARCHAR(50)` | Full name of the employee   |
| dept_name  | `VARCHAR(50)` | Department (e.g., IT, HR)   |
| salary     | `INT`       | Salary in local currency     |

---

## Key SQL Concepts Demonstrated

- ✅ Table design & data insertion  
- ✅ Aggregate functions (`MAX`, `AVG`)  
- ✅ Grouping by department & employee  
- ✅ Window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`)  
- ✅ Salary comparison logic using CASE statements  
- ✅ Department-level ranking & salary progression insights

---

## Highlighted Use Cases

- Find highest salaries overall and per department  
- Rank employees within departments based on salary  
- Compare an employee’s salary with previous/next department members  
- Display the first N employees per department  
- HR analytics dashboards for identifying top talent or salary anomalies

---

## How to Use

1. Copy and run `employee_analysis.sql` in any SQL client (e.g., PostgreSQL, MySQL, SQLite).  
2. Review and customize queries based on use case (e.g., bonus eligibility, performance tiering).  
3. Pair with BI tools like Tableau or Power BI for visualization.

---

## Ideal For

- Technical SQL interview prep  
- Portfolio demonstration for data analyst or BI roles  
- Learning window functions through HR-related datasets

---

> *This project is part of [Uday's Portfolio](https://github.com/udayrana11/Uday_Portfolio). Check out other analytical SQL projects there.*
