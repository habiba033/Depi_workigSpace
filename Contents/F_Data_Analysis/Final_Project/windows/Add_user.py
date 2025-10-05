import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
from datetime import datetime

# add root folder to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from db import save_to_db
from utils import show_message

class AddUserWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "add_user.ui")
        uic.loadUi(ui_path, self)
        self.parent = parent
        self.birthYearInput.setPlaceholderText("Enter Your Birth Year :")
        self.ageInput.setPlaceholderText("Enter Your Age :")
        
        # Connect buttons
        self.addUserButton.clicked.connect(self.add_user)
        self.cancelButton.clicked.connect(self.close)

#def to_save
    def add_user(self):
        """Save user when 'Add User' is clicked"""
        birth_year_text = self.birthYearInput.text().strip()
        age_text = self.ageInput.text().strip()
        gender = self.genderCombo.currentText()
        user_type = self.userTypeCombo.currentText()
        
        # Check empty fields
        if not birth_year_text or not age_text or not gender or not user_type:
            show_message(self, "Missing Data", "Please fill in all fields ❌", success=False)
            return
        
        #  Check if numbers
        if not birth_year_text.isdigit() or not age_text.isdigit():
            show_message(self, "Invalid Data", "Birth year and age must be numbers ❌", success=False)
            return

        birth_year = int(birth_year_text)
        age = int(age_text)
        current_year = datetime.now().year

        # Check logical birth year
        if birth_year < 1900 or birth_year > current_year:
            show_message(self, "Invalid Year", f"Birth year must be between 1900 and {current_year} ❌", success=False)
            return

        # Check consistency between birth year and age
        calculated_age = current_year - birth_year
        if abs(calculated_age - age) > 2:
            show_message(self, "Invalid Age", "Age does not match birth year ❌", success=False)
            return
        
        # Save to DB
        if save_to_db(birth_year, age, gender, user_type):
            show_message(self, "Success", "User added successfully ✅", success=True)
            self.birthYearInput.clear()
            self.ageInput.clear()
            self.genderCombo.setCurrentIndex(0)
            self.userTypeCombo.setCurrentIndex(0)
            
            # Refresh main window list
            self.close()
        else:
            show_message(self, "Error", "Could not save user ❌", success=False)
