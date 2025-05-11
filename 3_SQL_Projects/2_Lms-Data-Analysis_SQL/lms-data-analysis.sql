/* ============================================================
   File: lms_analysis.sql
   Author: Uday Rana
   Description: 
      This SQL file contains a set of optimized queries designed to 
      analyze and extract insights from a Learning Management System (LMS) 
      database. These queries provide detailed reports on various key 
      performance metrics, including session ratings, batch attendance, 
      student performance, and test scores, aimed at supporting data-driven 
      decision-making processes in educational environments.
   
      Key Metrics Analyzed:
      - Student ratings for sessions
      - Attendance statistics and performance at the batch level
      - Average test scores and pass rates
      - Student participation insights

   ============================================================ */

/* ================================
   1. Average Session Rating per Student
   Description: 
      - Calculates the average rating provided by each student for 
        the sessions they attended.
      - Returns session details including the instructor’s ID and 
        session ID, facilitating detailed insights on session quality 
        across different instructors.
================================== */
SELECT 
    a.student_id,
    AVG(a.rating) AS average_rating,
    s.conducted_by,          -- Instructor ID
    s.id AS session_id       -- Session ID
FROM attendances a
JOIN sessions s ON a.session_id = s.id
GROUP BY a.student_id, s.conducted_by, s.id;


/* ================================
   2. Retrieve All Records from Main Tables
   Description: 
      - Quick diagnostic queries to review data from the main LMS tables 
        for troubleshooting, data auditing, or quick insights.
================================== */
SELECT * FROM batches;
SELECT * FROM sessions;
SELECT * FROM users;
SELECT * FROM attendances;


/* ================================
   3. Batch Attendance Percentage Calculation
   Description:
      - Calculates the attendance percentage for each session within 
        a batch, showing the rate of participation per student.
      - Helps evaluate batch engagement and session reach.
      - Query optimizations ensure accurate session and attendance tracking.
================================== */
SELECT 
    b.id AS batch_id,
    u.name AS student_name,
    a.session_id,
    (COUNT(a.student_id) * 100.0 / COUNT(a.session_id)) AS percent_attendance
FROM batches b
JOIN sessions s ON b.id = s.batch_id
JOIN attendances a ON s.id = a.session_id
JOIN users u ON u.id = a.student_id
GROUP BY b.id, u.name, a.session_id;


/* ================================
   4. Session Attendance Percentage
   Description:
      - Computes the attendance percentage for each session by comparing 
        the number of students attending to the total active students 
        enrolled in the session's batch.
      - Provides key insights into overall session engagement.
================================== */
SELECT
    s.id AS session_id,
    b.name AS batch_name,
    s.conducted_by,         -- Instructor's ID
    COUNT(DISTINCT a.student_id) * 100.0 / COUNT(DISTINCT sbm.user_id) AS attendance_percentage
FROM sessions s
JOIN batches b ON s.batch_id = b.id
JOIN users u ON s.conducted_by = u.id
JOIN student_batch_maps sbm ON sbm.batch_id = b.id AND sbm.active = TRUE
LEFT JOIN attendances a ON a.session_id = s.id AND a.student_id = sbm.user_id
GROUP BY s.id, b.name, s.conducted_by;


/* ================================
   5. Average Test Score per Student
   Description:
      - Computes the average test score for each student, providing 
        insights into academic performance across multiple tests.
      - Facilitates the identification of underperforming students and 
        helps prioritize academic interventions.
================================== */
SELECT 
    u.id AS student_id,
    u.name AS student_name,
    AVG(ts.score) AS average_score
FROM test_scores ts
JOIN users u ON ts.user_id = u.id
JOIN student_batch_maps sbm ON sbm.user_id = u.id
GROUP BY u.id, u.name;


/* ================================
   6. Students Passing Each Test
   Description:
      - Determines the count of students who passed each test, where 
        passing is defined as scoring greater than 40% of the total marks.
      - Offers insight into test difficulty and overall academic success.
================================== */
SELECT 
    t.id AS test_id,
    COUNT(*) AS students_passed
FROM tests t
JOIN test_scores ts ON t.id = ts.test_id
WHERE (ts.score * 100.0) / t.total_mark > 40
GROUP BY t.id;


/* ================================
   7. Detailed Attendance Percentage per Student
   Description:
      - Calculates each student's attendance percentage relative to 
        the total sessions in their active batch.
      - The subquery 'total_sessions' computes the number of sessions 
        available for each student, ensuring that the attendance rate 
        is based on active participation in the batch.
      - Students mapped to 5 or fewer batches are considered.
      - Provides deep insights into the attendance performance of individual students.
================================== */
SELECT
    u.id AS student_id,
    u.name AS student_name,
    COUNT(a.session_id)::FLOAT / NULLIF(total_sessions.total, 0) * 100 AS attendance_percentage
FROM users u
JOIN student_batch_maps sbm ON u.id = sbm.user_id
JOIN batches b ON sbm.batch_id = b.id
LEFT JOIN attendances a ON a.student_id = u.id
LEFT JOIN sessions s ON s.id = a.session_id AND s.batch_id = sbm.batch_id
-- Total sessions for the active batch of each student
LEFT JOIN (
    SELECT
        sbm.user_id,
        COUNT(s.id) AS total
    FROM student_batch_maps sbm
    JOIN sessions s ON s.batch_id = sbm.batch_id
    WHERE sbm.active = TRUE
    GROUP BY sbm.user_id
) total_sessions ON total_sessions.user_id = u.id
WHERE sbm.active = TRUE
  AND (
       SELECT COUNT(*) 
       FROM student_batch_maps sbm2 
       WHERE sbm2.user_id = u.id
      ) <= 5
GROUP BY u.id, u.name, total_sessions.total;

