import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from db import delete_user_from_db
from utils import show_message

# ---------- delete Users Window ----------
class DeleteUserWindow(QWidget):
    def __init__(self):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "delete_user.ui")
        uic.loadUi(ui_path, self)
        
        #connect Buttons
        self.deleteButton.clicked.connect(self.delete_user)
        self.exitButton.clicked.connect(self.close)
        self.searchInput.setPlaceholderText("ID You Want To Delete: ")
    
#delte_from db
    def delete_user(self):
        user_id_text = self.searchInput.text().strip()

        if not user_id_text.isdigit():
            show_message(self, "Invalid ID", "Please enter a valid numeric User ID ❌", success=False)
            return

        user_id = int(user_id_text)

        # confirmation
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm Delete")
        msg.setText(f"Are you sure you want to delete user with ID {user_id}?")
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
            if delete_user_from_db(user_id):
                show_message(self, "Success", f"User ID {user_id} deleted successfully ✅", success=True)
                self.searchInput.clear()
            else:
                show_message(self, "Error", f"User ID {user_id} not found ❌", success=False)
