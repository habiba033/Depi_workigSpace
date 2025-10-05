import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
from datetime import datetime

# add root folder to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from db import get_connection
from db import save_trip_to_db
from utils import show_message   

# ---------- Add Trip Window ----------
class AddTripWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "add_trip.ui")
        uic.loadUi(ui_path, self)
        self.parent = parent
        
        # Load users and stations into comboboxes
        self.load_users()
        self.load_stations()

        # Connect save button
        self.saveButton.clicked.connect(self.save_trip)
        self.cancelButton.clicked.connect(self.close)

    def load_users(self):
        """Load all users into the ComboBox"""
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM dim_user ORDER BY user_id")
            users = cur.fetchall()
            cur.close()
            conn.close()

            self.userCombo.clear()
            # placeholder
            self.userCombo.addItem("Select a User...")
            self.userCombo.model().item(0).setEnabled(False)

            for user in users:
                self.userCombo.addItem(str(user[0]))

        except Exception as e:
            print("Error loading users:", e)

    def load_stations(self):
        """Load all stations into the ComboBoxes"""
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT station_id, station_name FROM dim_station ORDER BY station_id")
            stations = cur.fetchall()
            cur.close()
            conn.close()

            self.startStationCombo.clear()
            self.endStationCombo.clear()

            # placeholders
            self.startStationCombo.addItem("Select Start Station...")
            self.startStationCombo.model().item(0).setEnabled(False)

            self.endStationCombo.addItem("Select End Station...")
            self.endStationCombo.model().item(0).setEnabled(False)

            for station_id, station_name in stations:
                display_text = f"{station_id} - {station_name}"
                self.startStationCombo.addItem(display_text, station_id)
                self.endStationCombo.addItem(display_text, station_id)

        except Exception as e:
            print("Error loading stations:", e)


    def save_trip(self):
        try:
            # placeholder checks (you already have these)
            if self.userCombo.currentIndex() == 0 or \
                self.startStationCombo.currentIndex() == 0 or \
                self.endStationCombo.currentIndex() == 0:
                show_message(self, "Error", "Please select valid user and stations ❌", success=False)
                return

            user_id = int(self.userCombo.currentText())
            start_station_id = self.startStationCombo.currentData()
            end_station_id = self.endStationCombo.currentData()

            start_time = self.startTimeEdit.dateTime().toPyDateTime()
            end_time = self.endTimeEdit.dateTime().toPyDateTime()

            # call central DB function
            success, message = save_trip_to_db(user_id, start_station_id, end_station_id, start_time, end_time)

            if success:
                show_message(self, "Success", message, success=True)
                self.close()
            else:
                show_message(self, "Error", message, success=False)

        except Exception as e:
            show_message(self, "Error", f"Unexpected error: {e}", success=False)
