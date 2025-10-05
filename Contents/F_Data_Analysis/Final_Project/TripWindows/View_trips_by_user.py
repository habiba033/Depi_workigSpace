import sys
import os
from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import QWidget, QTableWidgetItem, QMessageBox, QHeaderView

# add root folder to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from db import get_connection


class ViewTripsByUserWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "view_trips_by_user.ui")
        uic.loadUi(ui_path, self)

        # setup table (added User ID column → total 8 columns)
        self.tripsTable.setColumnCount(8)
        self.tripsTable.setHorizontalHeaderLabels([
            "Trip ID", "Start Time", "End Time", "Duration (sec)",
            "Bike ID", "Start Station", "End Station", "User ID"
        ])
        header = self.tripsTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # search placeholder
        self.searchInput.setPlaceholderText("Enter User ID to view trips...")
        self.searchInput.textChanged.connect(self.search_trips)

        # load nothing at start
        self.tripsTable.setRowCount(0)

    def show_in_table(self, rows):
        """show trips in table"""
        self.tripsTable.setRowCount(0)
        for row in rows:
            row_idx = self.tripsTable.rowCount()
            self.tripsTable.insertRow(row_idx)
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value))

                # highlight User ID column
                if col_idx == 7:  # index of "User ID"
                    item.setBackground(QtGui.QColor("#afffa2"))  
                    font = QtGui.QFont()
                    font.setBold(True)
                    item.setFont(font)

                self.tripsTable.setItem(row_idx, col_idx, item)

    def search_trips(self, text):
        """search trips just by writing user_id"""
        search_term = text.strip()

        try:
            conn = get_connection()
            cur = conn.cursor()

            if search_term == "":
                # لو فاضية: نعرض كل الرحلات
                cur.execute("""
                    SELECT trip_id, start_time, end_time, duration_sec, bike_id,
                           start_station_id, end_station_id, user_id
                    FROM fact_trips ORDER BY trip_id
                """)
            elif search_term.isdigit():
                # بحث بالـ user_id
                cur.execute("""
                    SELECT trip_id, start_time, end_time, duration_sec, bike_id,
                           start_station_id, end_station_id, user_id
                    FROM fact_trips WHERE user_id = %s ORDER BY trip_id
                """, (search_term,))
            else:
                # لو كتب أي حاجة تانية: ignore
                self.tripsTable.setRowCount(0)
                return

            rows = cur.fetchall()
            cur.close()
            conn.close()

            self.show_in_table(rows)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Search failed: {e}")
