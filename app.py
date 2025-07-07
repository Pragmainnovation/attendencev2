# Streamlit Attendance App with Face Recognition and Location Check
# Admin login: admin / password

import streamlit as st
import pandas as pd
import cv2
from geopy.distance import geodesic
import os
from datetime import datetime
import numpy as np
from streamlit_js_eval import streamlit_js_eval
import json
from PIL import Image
from io import BytesIO
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
import hashlib
import logging

# Load face models using official pretrained weights (no .pth files needed)
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Company location
COMPANY_LOCATION = (30.022353800687085, 31.46491236103251)
MAX_DISTANCE_METERS = 600

# Paths
DATA_DB = 'data.db'  # New database for employees and attendance

# --- New DB setup and migration ---
def get_data_db_connection():
    conn = sqlite3.connect(DATA_DB, check_same_thread=False)
    return conn

def setup_data_db():
    conn = get_data_db_connection()
    c = conn.cursor()
    # Employees table: name (unique), face_encoding (JSON string)
    c.execute('''CREATE TABLE IF NOT EXISTS employees (
        name TEXT PRIMARY KEY,
        face_encoding TEXT NOT NULL
    )''')
    # Attendance table
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date TEXT,
        time TEXT,
        latitude REAL,
        longitude REAL,
        type TEXT,
        status TEXT,
        hour TEXT
    )''')
    conn.commit()
    conn.close()

setup_data_db()

# Database setup
def get_db_connection():
    conn = sqlite3.connect('attendance.db', check_same_thread=False)
    return conn

def setup_admin_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS admins (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

setup_admin_db()

def get_admins():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username, password FROM admins')
    admins = [{'username': row[0], 'password': row[1]} for row in c.fetchall()]
    conn.close()
    return admins

def add_admin(username, password):
    # Always store password as SHA-256 hash
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO admins (username, password) VALUES (?, ?)', (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def change_admin_password(username, old_password, new_password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT password FROM admins WHERE username=?', (username,))
    row = c.fetchone()
    if row and row[0] == old_password:
        c.execute('UPDATE admins SET password=? WHERE username=?', (new_password, username))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def remove_admin(username):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('DELETE FROM admins WHERE username=?', (username,))
    conn.commit()
    conn.close()

# Load admin credentials from file
CREDENTIALS_PATH = 'admin_credentials.json'
def load_admin_credentials():
    with open(CREDENTIALS_PATH, 'r') as f:
        creds = json.load(f)
    # Support both old and new format
    if isinstance(creds, dict) and 'admins' in creds:
        return creds['admins']
    elif isinstance(creds, dict) and 'username' in creds and 'password' in creds:
        return [{'username': creds['username'], 'password': creds['password']}]
    else:
        return []

def save_admin_credentials(admins):
    with open(CREDENTIALS_PATH, 'w') as f:
        json.dump({'admins': admins}, f)

# Helper functions
def load_employees():
    conn = get_data_db_connection()
    df = pd.read_sql_query('SELECT * FROM employees', conn)
    conn.close()
    return df

def save_employees(df):
    conn = get_data_db_connection()
    c = conn.cursor()
    c.execute('DELETE FROM employees')
    for _, row in df.iterrows():
        c.execute('INSERT OR REPLACE INTO employees (name, face_encoding) VALUES (?, ?)', (row['name'], row['face_encoding']))
    conn.commit()
    conn.close()

def load_attendance():
    conn = get_data_db_connection()
    df = pd.read_sql_query('SELECT * FROM attendance', conn)
    conn.close()
    return df

def save_attendance(df):
    conn = get_data_db_connection()
    c = conn.cursor()
    c.execute('DELETE FROM attendance')
    for _, row in df.iterrows():
        c.execute('''INSERT INTO attendance (name, date, time, latitude, longitude, type, status, hour) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                row.get('name', ''),
                row.get('date', ''),
                row.get('time', ''),
                row.get('latitude', None),
                row.get('longitude', None),
                row.get('type', ''),
                row.get('status', ''),
                str(row.get('hour', ''))
            ))
    conn.commit()
    conn.close()

def check_location(user_loc):
    dist = geodesic(COMPANY_LOCATION, user_loc).meters
    return dist <= MAX_DISTANCE_METERS, dist

def to_rgb_uint8(img_file_or_bytes):
    # Accepts file-like or bytes, returns RGB uint8 numpy array or None
    try:
        if hasattr(img_file_or_bytes, 'getvalue'):
            img_bytes = img_file_or_bytes.getvalue()
        elif isinstance(img_file_or_bytes, bytes):
            img_bytes = img_file_or_bytes
        else:
            return None
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        arr = np.array(pil_img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    except Exception:
        return None

# Custom embedding extraction

def extract_embedding_from_array(img_array):
    img = Image.fromarray(img_array).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding.squeeze(0).cpu().numpy()

def identify_person_custom(embedding, threshold=0.8):
    df = load_employees()
    best_name, best_dist = "Unknown", float('inf')
    for _, row in df.iterrows():
        encoding_list = row.get('face_encoding', None)
        if not encoding_list or encoding_list in ['nan', 'NaN', '']:
            continue
        try:
            stored_embeddings = np.array(json.loads(encoding_list))
            if stored_embeddings.ndim == 1:
                stored_embeddings = np.expand_dims(stored_embeddings, axis=0)
        except Exception:
            continue
        dists = np.linalg.norm(stored_embeddings - embedding, axis=1)
        min_dist = np.min(dists)
        if min_dist < best_dist:
            best_dist, best_name = min_dist, row['name']
    return best_name if best_dist < threshold else "Unknown"

# Logging setup
LOG_FILE = 'admin_actions.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(message)s')

def log_admin_action(action, username=None, details=None):
    # Try to get username from session if not provided
    if username is None:
        username = st.session_state.get('admin_username') or st.session_state.get('username')
    msg = f"[ADMIN ACTION]"
    if username:
        msg += f" User: {username} -"
    msg += f" {action}"
    if details:
        msg += f" | Details: {details}"
    logging.info(msg)

def get_log_file_contents():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Streamlit app
def main():
    # Rerun workaround: check for rerun flag
    if 'do_rerun' in st.session_state and st.session_state.do_rerun:
        st.session_state.do_rerun = False
        st.rerun()

    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    st.title('Employee Attendance System')

    # Sidebar: show logout if admin, else menu
    if st.session_state.admin_logged_in:
        if st.sidebar.button('Logout'):
            st.session_state.admin_logged_in = False
            st.session_state.do_rerun = True
        admin_dashboard()
    else:
        menu = ['Login', 'Attendance']
        choice = st.sidebar.selectbox('Menu', menu)
        if choice == 'Login':
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            if st.button('Login'):
                admins = get_admins()
                password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
                if any(u['username'] == username and u['password'] == password_hash for u in admins):
                    st.session_state.admin_logged_in = True
                    st.session_state.do_rerun = True
                else:
                    st.error('Invalid credentials')
        else:
            attendance_page()

def admin_dashboard():
    st.title('Admin Dashboard')
    # Attendance download button (from database)
    att_df = load_attendance()
    if st.download_button("Download Attendance CSV", att_df.to_csv(index=False), file_name="attendance.csv", mime="text/csv", key="download_attendance_csv"):
        log_admin_action("Downloaded attendance CSV", username=st.session_state.get('admin_username'))
    # Log file download button
    log_contents = get_log_file_contents()
    st.download_button("Download Log File", log_contents, file_name="admin_actions.log", mime="text/plain", key="download_log_file")
    # Remove employee option
    employees = load_employees()
    if not employees.empty:
        st.subheader("Remove Employee")
        employee_names = employees['name'].tolist()
        selected_remove = st.selectbox("Select employee to remove", employee_names)
        if st.button("Remove Employee"):
            employees = employees[employees['name'] != selected_remove]
            save_employees(employees)
            st.success(f"Employee '{selected_remove}' removed successfully!")
            log_admin_action("Removed employee", username=st.session_state.get('admin_username'), details=selected_remove)
    # Remove admin option
    st.subheader("Remove Admin")
    admins = get_admins()
    admin_usernames = [a['username'] for a in admins]
    if 'admin_logged_in' in st.session_state and st.session_state.get('admin_logged_in', False):
        current_admin = None
        # Try to get the currently logged-in admin username
        if 'admin_logged_in' in st.session_state and hasattr(st.session_state, 'admin_logged_in'):
            # Not storing username in session, so allow removal of any admin
            pass
    else:
        current_admin = None
    if admin_usernames:
        selected_admin_remove = st.selectbox("Select admin to remove", admin_usernames)
        if st.button("Remove Selected Admin"):
            if selected_admin_remove == 'admin':
                st.warning("Cannot remove the default 'admin' user for safety.")
            else:
                remove_admin(selected_admin_remove)
                st.success(f"Admin '{selected_admin_remove}' removed successfully!")
                log_admin_action("Removed admin", username=selected_admin_remove)
    # Add new employee form
    with st.form("add_employee_form"):
        new_name = st.text_input("Employee Name")
        method = st.radio("Add face by:", ["Upload Image(s)", "Use Camera"])
        uploaded_images = None
        camera_images = []
        if method == "Upload Image(s)":
            uploaded_images = st.file_uploader('Upload at least 5 Employee Face Images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        else:
            st.info("Capture 5 images for best recognition. Capture one, then click again for the next.")
            for i in range(5):
                img = st.camera_input(f'Capture Image {i+1}')
                camera_images.append(img)
        submitted = st.form_submit_button("Add New Employee")
        if submitted:
            if not new_name.strip():
                st.warning("Please enter a valid name.")
            elif (method == "Upload Image(s)" and (not uploaded_images or len(uploaded_images) < 5)) or (method == "Use Camera" and (not all(camera_images))):
                st.warning("Please provide at least 5 images.")
            else:
                face_found = False
                embeddings = []
                images_to_process = []
                if method == "Upload Image(s)":
                    for img_file in uploaded_images:
                        rgb_img = to_rgb_uint8(img_file)
                        if rgb_img is not None:
                            images_to_process.append(rgb_img)
                else:
                    for img in camera_images:
                        rgb_img = to_rgb_uint8(img)
                        if rgb_img is not None:
                            images_to_process.append(rgb_img)
                for rgb_img in images_to_process:
                    embedding = extract_embedding_from_array(rgb_img)
                    if embedding is not None:
                        face_found = True
                        embeddings.append(embedding.tolist())
                if not face_found or len(embeddings) < 5:
                    st.warning("At least 5 valid face images are required. Please try again.")
                else:
                    employees = load_employees()
                    if new_name in employees["name"].values:
                        st.warning("Employee already exists.")
                    else:
                        # Store all embeddings as a list
                        encoding_str = json.dumps(embeddings)
                        new_row = pd.DataFrame([{ "name": new_name, "face_encoding": encoding_str }])
                        employees = pd.concat([employees, new_row], ignore_index=True)
                        save_employees(employees)
                        st.success(f"Employee '{new_name}' added successfully!")
                        log_admin_action("Added new employee", username=st.session_state.admin_username, details=new_name)
    # Change password section
    st.subheader("Change Admin Password")
    with st.form("change_password_form"):
        username = st.text_input("Admin Username")
        old_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm = st.form_submit_button("Change Password")
        if confirm:
            if change_admin_password(username, old_password, new_password):
                st.success("Password changed successfully!")
                log_admin_action("Changed password", username=username)
            else:
                st.error("Invalid username or current password.")
    # Add new admin section
    st.subheader("Add New Admin")
    with st.form("add_admin_form"):
        new_admin_user = st.text_input("New Admin Username")
        new_admin_pass = st.text_input("New Admin Password", type="password")
        add_admin_btn = st.form_submit_button("Add Admin")
        if add_admin_btn:
            if add_admin(new_admin_user, new_admin_pass):
                st.success(f"Admin '{new_admin_user}' added successfully!")
                log_admin_action("Added new admin", username=new_admin_user)
            else:
                st.warning("Admin username already exists.")
    # View and download logs
    st.subheader("Admin Actions Log")
    log_contents = get_log_file_contents()
    if log_contents:
        st.text_area("Log Contents", value=log_contents, height=300)
        # Download log button
        if st.download_button("Download Log File", log_contents, file_name="admin_actions.log", mime="text/plain"):
            log_admin_action("Downloaded log file", username=st.session_state.get('admin_username'))
    else:
        st.info("No logs found.")

def attendance_page():
    st.header('Employee Attendance')
    st.warning(
        "On iPhone/iPad, please allow location access when prompted. "
        "If you denied it, go to Settings > Safari > Location, enable it for this site, and refresh the page. "
        "For best results, use Chrome or Firefox instead of Safari on iOS."
    )

    # Initialize session state variables
    if 'location_requested' not in st.session_state:
        st.session_state['location_requested'] = False
    if 'location_start_time' not in st.session_state:
        st.session_state['location_start_time'] = None

    # Only Get Location button
    if st.button('Get Location'):
        st.session_state['location_requested'] = True
        st.session_state['location_start_time'] = datetime.now().timestamp()

    latitude, longitude = None, None
    if st.session_state['location_requested']:
        js_code = """
        new Promise((resolve, reject) => {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        resolve({
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude
                        });
                    },
                    (error) => {
                        resolve({latitude: null, longitude: null, error: error.message});
                    }
                );
            } else {
                resolve({latitude: null, longitude: null, error: 'Geolocation not supported'});
            }
        });
        """
        loc = streamlit_js_eval(js_expressions=js_code, key="get_location")
        latitude = loc.get('latitude') if loc else None
        longitude = loc.get('longitude') if loc else None
        if latitude and longitude:
            st.success(f"Detected location: {latitude}, {longitude}")
        else:
            elapsed = datetime.now().timestamp() - st.session_state['location_start_time']
            st.warning('Waiting for location permission or location not available.')
            if elapsed > 10:
                st.error('Location detection timed out. Please try again.')
    else:
        st.info('Click "Get Location" to request location access.')

    captured_image = st.camera_input('Take a photo')
    recognized_name = None
    recognition_error = None
    allow_submit = False
    if captured_image:
        rgb_img = to_rgb_uint8(captured_image)
        if rgb_img is None:
            st.error("Captured image is not a valid color image. Please use a standard JPG or PNG.")
            recognition_error = 'Invalid image format.'
        else:
            embedding = extract_embedding_from_array(rgb_img)
            if embedding is not None:
                employees = load_employees()
                try:
                    recognized_name = identify_person_custom(embedding)
                except Exception as e:
                    recognition_error = f'Face recognition error: {e}'
                if recognized_name == "Unknown":
                    recognition_error = 'Face not recognized. Please try again with a clearer photo.'
                    st.error(recognition_error)
                    allow_submit = False
                else:
                    allow_submit = True
            else:
                recognition_error = 'No face detected in image.'
                st.error(recognition_error)
                allow_submit = False
    if captured_image and recognized_name and recognized_name != "Unknown":
        st.success(f"Recognized: {recognized_name}")

    # Require both location and face recognition for submit
    location_ready = latitude is not None and longitude is not None
    submit_disabled = not (captured_image and allow_submit and location_ready)

    if st.button('Submit Attendance', disabled=submit_disabled):
        user_loc = (latitude, longitude)
        is_near, dist = check_location(user_loc)
        if is_near:
            # Query the database for all records for this employee, fetch unsorted
            conn = get_data_db_connection()
            c = conn.cursor()
            c.execute('''SELECT type, date, time FROM attendance WHERE name = ?''', (recognized_name,))
            rows = c.fetchall()
            conn.close()
            last_type = None
            last_checkin_dt = None
            if rows:
                # Build a list of (type, datetime) and sort
                records = []
                for t, d, tm in rows:
                    try:
                        dt = pd.to_datetime(f'{d} {tm}', errors='coerce')
                        if pd.notnull(dt):
                            records.append((t, dt))
                    except Exception:
                        continue
                if records:
                    records.sort(key=lambda x: x[1], reverse=True)  # Most recent first
                    last_type = records[0][0]
                    if last_type == 'Check In':
                        for t, dt in records:
                            if t == 'Check In':
                                last_checkin_dt = dt
                                break
            now = datetime.now()
            now_time = now.time()
            hour_value = ""
            if last_type == 'Check In':
                check_type = 'Check Out'
                status = 'early leave' if now_time < datetime.strptime('19:00', '%H:%M').time() else ''
                if last_checkin_dt is not None:
                    hour_value = round((now - last_checkin_dt).total_seconds() / 3600, 2)
            else:
                check_type = 'Check In'
                status = 'late' if now_time > datetime.strptime('12:00', '%H:%M').time() else ''
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')
            new_row = pd.DataFrame([{
                'name': recognized_name,
                'date': date_str,
                'time': time_str,
                'latitude': latitude,
                'longitude': longitude,
                'type': check_type,
                'status': status,
                'hour': hour_value
            }])
            att_df = load_attendance()
            if 'datetime' in att_df.columns:
                att_df = att_df.drop(columns=['datetime'])
            for col in ['date', 'time', 'status', 'hour']:
                if col not in att_df.columns:
                    att_df[col] = ""
            att_df = pd.concat([att_df, new_row], ignore_index=True)
            save_attendance(att_df)
            st.success(f'{check_type} marked for {recognized_name}!')
            st.session_state['location_requested'] = False
            st.session_state['location_start_time'] = None
            st.balloons()
        else:
            st.error(f'You are too far from the company location ({dist:.2f} meters).')

if __name__ == '__main__':
    main()
