import sys
import os
from PyQt5 import uic
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget,QTableWidgetItem
from PyQt5.QtWidgets import QHeaderView

# أضف root folder إلى sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from db import get_connection
from utils import QMessageBox

#  ---------- View Users Window ----------

class ViewStationsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "view_stations.ui")
        uic.loadUi(ui_path, self)

        
        self.stationsTable.setColumnCount(4)
        self.stationsTable.setHorizontalHeaderLabels(["ID", "Station Name", "Latitude", "Longitude"])
        header = self.stationsTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.searchInput.setPlaceholderText("Search By ID: ")
        self.searchInput.textChanged.connect(self.search_station)

        #load all stations when opening the page
        self.load_all_stations()

    def load_all_stations(self):
        """view all stations in the table"""
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT station_id, station_name, latitude, longitude FROM dim_station ORDER BY station_id")
            rows = cur.fetchall()
            cur.close()
            conn.close()

            self.show_in_table(rows)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load stations: {e}")

    def show_in_table(self, rows):
        """show data in the table"""
        self.stationsTable.setRowCount(0)
        for row in rows:
            row_idx = self.stationsTable.rowCount()
            self.stationsTable.insertRow(row_idx)
            for col_idx, value in enumerate(row):
                self.stationsTable.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    def search_station(self, text):
        """search just by writing"""
        search_term = text.strip()

        try:
            conn = get_connection()
            cur = conn.cursor()

            if search_term == "":
                # if search is empty return all data
                cur.execute("SELECT station_id, station_name, latitude, longitude FROM dim_station ORDER BY station_id")
            elif search_term.isdigit():
                # search by ID 
                cur.execute("SELECT station_id, station_name, latitude, longitude FROM dim_station WHERE station_id = %s", (search_term,))
            else:
                # search by station name 
                cur.execute("SELECT station_id,station_name, latitude, longitude FROM dim_station WHERE station_name ILIKE %s",
                            (f"%{search_term}%,"))

            rows = cur.fetchall()
            cur.close()
            conn.close()

            self.show_in_table(rows)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Search failed: {e}")
