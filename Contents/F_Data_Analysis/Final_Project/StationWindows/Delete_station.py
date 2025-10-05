import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from db import delete_station_from_db
from utils import show_message

# ---------- delete Users Window ----------
class DeleteStationWindow(QWidget):
    def __init__(self):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "delete_station.ui")
        uic.loadUi(ui_path, self)
        
        #connect Buttons
        self.deleteButton.clicked.connect(self.delete_station)
        self.exitButton.clicked.connect(self.close)
        self.searchInput.setPlaceholderText("Station ID You Want To Delete: ")
    
#delete_station_from_db
    def delete_station(self):
        station_id_text = self.searchInput.text().strip()

        if not station_id_text.isdigit():
            show_message(self, "Invalid ID", "Please enter a valid numeric Station ID ❌", success=False)
            return

        station_id = int(station_id_text)

        # confirmation
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm Delete")
        msg.setText(f"Are you sure you want to delete station with ID {station_id}?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setWindowModality(Qt.ApplicationModal)
    # change the font type and color
        font = QFont("Arial", 12, QFont.Bold)
        msg.setFont(font)
        msg.setStyleSheet("QLabel{color: black;} QPushButton{min-width:80px;}")
    #execute
        reply = msg.exec_()

        if reply == QMessageBox.Yes:
            if delete_station_from_db(station_id):
                show_message(self, "Success", f"Station ID {station_id} deleted successfully ✅", success=True)
                self.searchInput.clear()
            else:
                show_message(self, "Error", f"Station ID {station_id} not found ❌", success=False)
