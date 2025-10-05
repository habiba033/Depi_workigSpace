import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow

# أضف المسار الجذري إلى sys.path
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

#------------ users -> windows --------------
from windows.Add_user import AddUserWindow
from windows.Delete_user import DeleteUserWindow
from windows.View_users import ViewUsersWindow

#------------ stations -> StationWindows
from StationWindows.Add_station import AddStationWindow
from StationWindows.Delete_station import DeleteStationWindow
from StationWindows.View_stations import ViewStationsWindow

#------------ trips -> StationWindows
from TripWindows.Add_trip import AddTripWindow
from TripWindows.Delete_trip import DeleteTripWindow
from TripWindows.View_trips_by_user import ViewTripsByUserWindow
from TripWindows.View_trips_by_station import ViewTripsByStartStationWindow
# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(base_dir, "ui", "main.ui")
        uic.loadUi(ui_path, self)
        
        # Users Signals
        self.addUserButton.clicked.connect(self.open_add_user)
        self.deleteButton.clicked.connect(self.open_delete_user)
        self.viewUsersButton.clicked.connect(self.open_view_users)
        
        # Stations Signals
        self.addStationButton.clicked.connect(self.open_add_station)
        self.SdeleteButton.clicked.connect(self.open_delete_station)
        self.viewStationsButton.clicked.connect(self.open_view_stations)
        
        # Trips Signals
        self.addTripButton.clicked.connect(self.open_add_trip)
        self.TdeleteButton.clicked.connect(self.open_delete_trip)
        self.viewTripsByUserButton.clicked.connect(self.open_view_trips_by_user)
        self.viewTripsByStationButton.clicked.connect(self.open_view_trips_by_station)
        
# ---------- USER ----------
    def open_add_user(self):
        self.add_user_window = AddUserWindow(self)
        self.add_user_window.show()
        
    def open_delete_user(self):
        self.delete_user_window = DeleteUserWindow()
        self.delete_user_window.show()
        
    def open_view_users(self):
        self.view_users_window = ViewUsersWindow()
        self.view_users_window.show()

# ---------- STATION ----------
    def open_add_station(self):
        self.add_station_window = AddStationWindow(self)
        self.add_station_window.show()
        
    def open_delete_station(self):
        self.delete_station_window = DeleteStationWindow()
        self.delete_station_window.show()

    def open_view_stations(self):
        self.view_stations_window= ViewStationsWindow()
        self.view_stations_window.show()
        
# ---------- Trips ----------
    def open_add_trip(self):
        self.add_trip_window = AddTripWindow(self)
        self.add_trip_window.show()
        
    def open_delete_trip(self):
        self.delete_trip_window = DeleteTripWindow()
        self.delete_trip_window.show()
        
    def open_view_trips_by_user(self):
        self.view_Utrips_window = ViewTripsByUserWindow()
        self.view_Utrips_window.show()
        
    def open_view_trips_by_station(self):
        self.view_Strips_window = ViewTripsByStartStationWindow()
        self.view_Strips_window.show()
        
    
# ---------- Run App ----------
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
    
    
    
