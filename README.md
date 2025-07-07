# Employee Attendance System

A Streamlit-based employee attendance system with face recognition and location verification.

## Features
- Employee check-in and check-out with face recognition
- Location verification (within 600 meters of company location)
- Late check-in and early leave detection
- Tracks hours spent at the company (on check-out)
- Admin dashboard for:
  - Adding/removing employees (requires 5 face images per employee)
  - Downloading attendance data as CSV
  - Downloading site action logs
  - Adding new admins
- Multiple admin support

## How It Works
- Employees check in/out by taking a photo and sharing their location.
- The system alternates between check-in and check-out based on the last attendance record for each employee.
- Attendance records include date, time, type (check in/out), status (late/early leave), and hours worked (on check-out).

## Setup
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Use the sidebar to log in as admin or access the attendance page.
- Admins can manage employees and other admins from the dashboard.
- Attendance and employee data are stored in a local SQLite database (`data.db`).
- You can download attendance records and site logs from the admin dashboard.

## Notes
- For best results, use Chrome or Firefox on desktop or Android. On iOS, allow location and camera access.
- Face recognition requires 5 clear face images per employee for best accuracy.

## Requirements
See `requirements.txt` for all dependencies.

---
Developed for secure, location-based employee attendance using facial recognition.
