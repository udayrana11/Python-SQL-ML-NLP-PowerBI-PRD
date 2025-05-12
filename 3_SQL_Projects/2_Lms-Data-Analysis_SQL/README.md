## Learning Management System (LMS) SQL Analysis

## Overview

This project contains a set of optimized SQL queries aimed at analyzing a Learning Management System (LMS) database. The queries are structured to extract meaningful insights about student performance, session ratings, attendance rates, and test outcomes. These insights can help educators and administrators make informed, data-driven decisions.

---

## 🧑‍💻 Author

**Uday Rana**  
📅 Date: May 2025  
🔗 Repository: *https://github.com/udayrana11/Uday_Portfolio*

---

## Tables Referenced

- `users`: User/student records
- `batches`: Grouped cohorts for learning
- `sessions`: Individual session records conducted by instructors
- `attendances`: Logs of student session attendance and ratings
- `tests`: Metadata about tests
- `test_scores`: Student scores in each test
- `student_batch_maps`: Mapping of students to their active or past batches

---

## Key Metrics Analyzed

| # | Focus Area |
|--:|------------|
| 1 | Average session rating per student |
| 2 | Attendance percentage per batch and session |
| 3 | Overall session attendance across batches |
| 4 | Average test score per student |
| 5 | Test-level pass counts |
| 6 | Individual student attendance percentages |
| 7 | Full-table diagnostics for LMS debugging |

---

## SQL Concepts Applied

- Aggregations (`AVG`, `COUNT`)
- Conditional filtering (`CASE WHEN`)
- Joins (inner, left, multi-table)
- Subqueries and derived tables
- Data type casting and null handling

---

## Query Highlights

### Session Quality Metrics
- **Avg. Session Rating per Student**: Determine how students rate sessions across instructors.
- **Session Attendance %**: Measure engagement by batch.

### Student Insights
- **Average Test Scores**: Identify high and low performers.
- **Test Pass Rates**: Reveal the number of students crossing the 40% threshold.
- **Attendance Ratios**: Drill into each student's engagement level.

### Full Table Previews
- Easy access to raw data via `SELECT *` for diagnostics from tables like `users`, `batches`, `sessions`, `attendances`.

---

## Use Cases

- Academic performance analysis
- Student retention strategy
- Attendance pattern recognition
- LMS system audits and QA
- Educator feedback evaluation

---

## How to Use

1. Load LMS data into a PostgreSQL (or compatible) database.
2. Run each query in sequence or use them as building blocks in BI tools like Power BI or Tableau.
3. Customize logic (e.g., pass threshold, batch filters) based on institutional rules.

---

> 📝 *Each query includes in-line documentation for clarity and modularity. Designed to showcase advanced SQL reporting skills with real-world LMS data structures.*
