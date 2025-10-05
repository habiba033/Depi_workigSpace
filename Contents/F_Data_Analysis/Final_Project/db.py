import psycopg2
from datetime import datetime
from psycopg2 import sql

# ---------- DB Connection ----------
def get_connection():
    return psycopg2.connect(
        dbname="GoBike",
        user="postgres",
        password="habiba2004#",
        host="localhost",
        port="5433"
    )


def save_to_db(birth_year, age, gender, user_type):
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO dim_user (birth_year, age, gender, user_type)
                VALUES (%s, %s, %s, %s)
            """, (birth_year, age, gender, user_type))
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print("Error saving user:", e)
            return False
        
        
def delete_user_from_db(user_id):
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM dim_user WHERE user_id = %s", (user_id,))
            affected_rows = cur.rowcount
            conn.commit()
            cur.close()
            conn.close()
            return affected_rows > 0  # True if removed
        except Exception as e:
            print("Error deleting user:", e)
            return False 
        
        
def save_station_to_db(station_name, latitude, longitude):
    """
    Save a station to the database.
    Returns True if saved successfully, False otherwise.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Insert station
        cur.execute("""
            INSERT INTO dim_station (station_name, latitude, longitude)
            VALUES (%s, %s, %s)
        """, (station_name, latitude, longitude))
        
        conn.commit()
        cur.close()
        conn.close()
        return True

    except Exception as e:
        print("Error saving station:", e)
        return False



def delete_station_from_db(station_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM dim_station WHERE station_id = %s", (station_id,))
        affected_rows = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return affected_rows > 0  # True if removed
    except Exception as e:
        print("Error deleting station:", e)
        return False 
    
def ensure_time_row(conn, dt):
    """
    Ensure a dim_time row exists for the given datetime (dt).
    Returns time_id.
    """
    cur = conn.cursor()
    # look by date + hour to be precise; adjust if you want different granularity
    cur.execute("SELECT time_id FROM dim_time WHERE date = %s AND hour = %s", (dt.date(), dt.hour))
    row = cur.fetchone()
    if row:
        time_id = row[0]
    else:
        day_of_week = dt.strftime("%A")  # e.g. "Monday"
        cur.execute("""
            INSERT INTO dim_time (date, hour, day, day_of_week, month, year)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING time_id
        """, (dt.date(), dt.hour, dt.day, day_of_week, dt.month, dt.year))
        time_id = cur.fetchone()[0]
        conn.commit()   # commit the inserted dim_time row
    cur.close()
    return time_id


def save_trip_to_db(user_id, start_station_id, end_station_id, start_dt, end_dt, bike_id=None):
    """
    Insert a trip into the data warehouse tables.
    Returns (True, message) on success or (False, error_message) on failure.
    """
    # validation
    if end_dt <= start_dt:
        return False, "End time must be after start time"

    duration_sec = int((end_dt - start_dt).total_seconds())
    if bike_id is None:
        bike_id = f"BIKE-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # ensure dim_time exists (and get time_id)
        time_id = ensure_time_row(conn, start_dt)

        # determine which fact table exists and insert
        # try fact_trip then fallback to fact_trips if not found
        for table_name in ("fact_trip", "fact_trips"):
            try:
                insert_q = sql.SQL("""
                    INSERT INTO {table} 
                    (start_time, end_time, duration_sec, bike_id, start_station_id, end_station_id, user_id, time_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """).format(table=sql.Identifier(table_name))

                cur.execute(insert_q, (
                    start_dt, end_dt, duration_sec, bike_id,
                    start_station_id, end_station_id, user_id, time_id
                ))
                conn.commit()
                cur.close()
                return True, f"Trip inserted into `{table_name}` successfully"
            except psycopg2.errors.UndefinedTable:
                # table does not exist; rollback and try next name
                conn.rollback()
                continue

        # if we reached here none of the expected tables existed
        return False, "No fact_trip(s) table exists (tried 'fact_trip' and 'fact_trips')"

    except Exception as e:
        if conn:
            conn.rollback()
        return False, str(e)
    finally:
        if conn:
            conn.close()
            

def delete_trip_from_db(trip_id):
    """Delete trip by trip_id from fact_trips"""
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM fact_trips WHERE trip_id = %s", (trip_id,))
        rows_deleted = cur.rowcount

        conn.commit()
        cur.close()
        conn.close()

        return rows_deleted > 0
    except Exception as e:
        print("Error deleting trip:", e)
        return False
