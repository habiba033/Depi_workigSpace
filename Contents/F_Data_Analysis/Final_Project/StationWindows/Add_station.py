import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
# add root folder to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from db import save_station_to_db
from utils import show_message


class AddStationWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "add_station.ui")
        print("llll",ui_path)
        uic.loadUi(ui_path, self)
        self.parent = parent
        self.stationNameInput.setPlaceholderText("Enter station name")
        self.latitudeInput.setPlaceholderText("Enter latitude in decimal degrees")
        self.longitudeInput.setPlaceholderText("Enter longitude in decimal degrees")
        
        
        # Connect buttons
        self.addStationButton.clicked.connect(self.add_station)
        self.cancelButton.clicked.connect(self.close)
        
    
    def add_station(self):
        station_name = self.stationNameInput.text().strip()
        latitude_text = self.latitudeInput.text().strip()
        longitude_text = self.longitudeInput.text().strip()
    
    # Check for empty fields
        if not station_name or not latitude_text or not longitude_text:
            show_message(self,"Missing Data", "Please fill in all fields❌ " ,success=False)
            return
    
    # Validate latitude and longitude as floats
        try:
            latitude = float(latitude_text)
            longitude = float(longitude_text)
        except ValueError:
            show_message(self,"Invalid Data", "Latitude and Longitude must be numbers ", success=False)
            return
    
    # check valid ranges
        if not (-90 <= latitude <= 90):
            show_message(self,"Invalid Latitude", "Latitude must be between -90 and 90 ", success=False)
            return
        if not (-180 <= longitude <= 180):
            show_message(self,"Invalid Longitude", "Longitude must be between -180 and 180 ", success=False)
            return
    
    #Save to DB
        if save_station_to_db(station_name, latitude, longitude):
            show_message(self,"Success", f"Station '{station_name}' added successfully ")
            self.stationNameInput.clear()
            self.latitudeInput.clear()
            self.longitudeInput.clear()
        else:
            show_message(self,"Error", "Could not save station ", success=False)
            
