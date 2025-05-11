/* ============================================================
   File         : employee_analysis.sql
   Author       : Uday Rana
   Description  : This SQL script contains the creation and
                  population of an Employee table, followed by 
                  queries demonstrating various SQL techniques 
                  including aggregation, window functions, and 
                  ranking mechanisms to analyze employee data.
   ============================================================ */

/* ===========================
   1. Create Employee Table
   =========================== */

-- The Employee table holds basic details of employees, including their ID, name, department, and salary.
CREATE TABLE employee (
    emp_Id INT PRIMARY KEY,            -- Employee ID (Primary Key)
    emp_name VARCHAR(50) NOT NULL,      -- Employee Name
    dept_name VARCHAR(50) NOT NULL,     -- Department Name
    salary INT NOT NULL                -- Employee Salary (in some currency)
);

/* ===========================
   2. Insert Data into Employee Table
   =========================== */

-- Insert sample data into the employee table to simulate real-world employee records.
INSERT INTO employee VALUES (101, 'Mohan', 'Admin', 4000);
INSERT INTO employee VALUES (102, 'Rajkumar', 'HR', 3000);
INSERT INTO employee VALUES (103, 'Akbar', 'IT', 4000);
INSERT INTO employee VALUES (104, 'Dorvin', 'Finance', 6500);
INSERT INTO employee VALUES (105, 'Rohit', 'HR', 3000);
INSERT INTO employee VALUES (106, 'Rajesh', 'Finance', 5000);
INSERT INTO employee VALUES (107, 'Preet', 'HR', 7000);
INSERT INTO employee VALUES (108, 'Maryam', 'Admin', 4000);
INSERT INTO employee VALUES (109, 'Sanjay', 'IT', 6500);
INSERT INTO employee VALUES (110, 'Vasudha', 'IT', 7000);
INSERT INTO employee VALUES (111, 'Melinda', 'IT', 8000);
INSERT INTO employee VALUES (112, 'Komal', 'IT', 10000);
INSERT INTO employee VALUES (113, 'Gautham', 'Admin', 2000);
INSERT INTO employee VALUES (114, 'Manisha', 'HR', 3000);
INSERT INTO employee VALUES (115, 'Chandni', 'IT', 4500);
INSERT INTO employee VALUES (116, 'Satya', 'Finance', 6500);
INSERT INTO employee VALUES (117, 'Adarsh', 'HR', 3500);
INSERT INTO employee VALUES (118, 'Tejaswi', 'Finance', 5500);
INSERT INTO employee VALUES (119, 'Cory', 'HR', 8000);
INSERT INTO employee VALUES (120, 'Monica', 'Admin', 5000);
INSERT INTO employee VALUES (121, 'Rosalin', 'IT', 6000);
INSERT INTO employee VALUES (122, 'Ibrahim', 'IT', 8000);
INSERT INTO employee VALUES (123, 'Vikram', 'IT', 8000);
INSERT INTO employee VALUES (124, 'Dheeraj', 'IT', 11000);

/* ============================================================
   3. Employee Salary Analysis Queries
   ============================================================ */

/* --- 3.1 Maximum Salary Overall ---
   Retrieve the highest salary across all employees in the database. 
   This query aggregates all salary records to find the maximum salary value. */
SELECT MAX(salary) AS max_salary
FROM employee;

/* --- 3.2 Maximum Salary per Employee in Each Department ---
   The query shows the maximum salary for each employee per department. 
   It is grouped by both `emp_name` and `dept_name` to ensure we get 
   the correct maximum salary value per employee in the department. */
SELECT emp_name, dept_name, MAX(salary) AS employee_salary
FROM employee
GROUP BY emp_name, dept_name;

/* --- 3.3 Maximum Salary by Department ---
   Retrieve the maximum salary in each department by grouping records 
   based on `dept_name`. This shows the highest salary within each department. */
SELECT dept_name, MAX(salary) AS max_salary_by_dept
FROM employee
GROUP BY dept_name;

/* --- 3.4 Display Each Employee Along with the Overall Maximum Salary 
         Within Their Department using Window Function ---
   This query uses the `MAX()` window function to calculate the maximum 
   salary within each department and display it alongside each employee's record. */
SELECT 
    emp_Id,
    emp_name,
    dept_name,
    salary,
    MAX(salary) OVER (PARTITION BY dept_name) AS overall_max_salary
FROM employee;

/* --- 3.5 Display Row Number for Each Department ---
   This query uses the `ROW_NUMBER()` window function to assign a unique 
   row number for each employee within their respective department. 
   Employees are ordered by their `emp_Id`. */
SELECT 
    e.*,
    ROW_NUMBER() OVER (PARTITION BY dept_name ORDER BY emp_Id) AS rn_by_dept
FROM employee AS e;

/* --- 3.6 Retrieve the First Two Employees (by emp_Id) per Department ---
   This query demonstrates how to select the first two employees by their 
   `emp_Id` in each department using the `ROW_NUMBER()` window function. */
SELECT * 
FROM (
    SELECT 
        e.*,
        ROW_NUMBER() OVER (PARTITION BY dept_name ORDER BY emp_Id) AS rn
    FROM employee AS e
) AS sub
WHERE rn < 3;

/* --- 3.7 Ranking Employees by Salary Within Each Department ---
         Using RANK, DENSE_RANK, and ROW_NUMBER ---
   This query shows three different ranking mechanisms:
   1. `RANK()` assigns a unique rank to employees based on salary within their department.
   2. `DENSE_RANK()` assigns ranks without gaps, even if multiple employees have the same salary.
   3. `ROW_NUMBER()` assigns a unique sequential number to each employee. */
SELECT 
    e.*,
    RANK() OVER (PARTITION BY dept_name ORDER BY salary DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY dept_name ORDER BY salary DESC) AS dense_rnk,
    ROW_NUMBER() OVER (PARTITION BY dept_name ORDER BY salary DESC) AS rn
FROM employee AS e;

/* --- 3.8 Using LAG and LEAD for Salary Comparison ---
         Get previous and following employee salary within each department ---
   The `LAG()` and `LEAD()` window functions are used here to compare the 
   current employee's salary with the previous and next employee's salary 
   in their respective department. */
SELECT 
    e.*,
    LAG(salary) OVER (PARTITION BY dept_name ORDER BY emp_Id) AS prev_employee_salary,
    LEAD(salary) OVER (PARTITION BY dept_name ORDER BY emp_Id) AS next_employee_salary
FROM employee AS e;

/* --- 3.9 LAG with Offset and a Salary Range Calculation ---
         Compare current employee's salary with the previous salary 
         in the same department ---
   This query calculates a salary range based on the current employee's 
   salary compared to the previous employee's salary, allowing for a 
   comparison with a lag of 1 and 2 (offset) rows within the department. */
SELECT 
    e.*,
    LAG(salary, 2, 0) OVER (PARTITION BY dept_name ORDER BY emp_Id) AS two_back_salary,
    LAG(salary) OVER (PARTITION BY dept_name ORDER BY emp_Id) AS prev_employee_salary,
    CASE 
        WHEN salary > LAG(salary) OVER (PARTITION BY dept_name ORDER BY emp_Id) THEN 'Higher than previous employee'
        WHEN salary = LAG(salary) OVER (PARTITION BY dept_name ORDER BY emp_Id) THEN 'Same as previous employee'
        WHEN salary < LAG(salary) OVER (PARTITION BY dept_name ORDER BY emp_Id) THEN 'Lower than previous employee'
    END AS salary_range
FROM employee AS e;

