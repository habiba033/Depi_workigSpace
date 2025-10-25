
import sys
from PyQt5 import QtWidgets, uic, QtCore
import joblib
import numpy as np
import pandas as pd

# --- ADDED: Imports from visuals_window.py ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QScrollArea, QSpacerItem, 
                             QSizePolicy, QGraphicsDropShadowEffect,
                             QGridLayout, QTabWidget, QTableWidget,
                             QTableWidgetItem)
from PyQt5.QtChart import (QChart, QChartView, QBarSeries, QBarSet, 
                           QValueAxis, QBarCategoryAxis, QLineSeries,
                           QSplineSeries, QHorizontalBarSeries)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QFont, QColor, QPalette, QPen

from predictor import PredictWindow
from visuals_window import VizWindow
# --- END ADDED IMPORTS ---
# --- ADDED: Stylesheet from visuals_window.py ---
# (I've also fixed a missing semicolon on 'width:200px' and an extra '}')

# --- END ADDED STYLESHEET ---


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        try:
            uic.loadUi(r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_2\QT5_DeskApp\ui\mainn_window.ui", self)
            self.btnQuickPredict.clicked.connect(self.open_predict_window)
            self.btn_nav_predict.clicked.connect(self.open_predict_window)
            
            
            # --- MODIFIED: Connect the visualize button ---
            # **IMPORTANT**: This assumes your .ui file has a button named 'btnVisualize'
            if hasattr(self, 'btnQuickVisual'):
                self.btnQuickVisual.clicked.connect(self.open_visualize_window)
            else:
                print("Warning: 'btnQuickVisual' not found in mainn_window.ui. Add it in Qt Designer.")
            if hasattr(self, 'btn_nav_visual'):
                self.btn_nav_visual.clicked.connect(self.open_visualize_window)
            else:
                print("Warning: 'btn_nav_visual' not found in mainn_window.ui. Add it in Qt Designer.")
                
                
            # --- END MODIFICATION ---

        except FileNotFoundError:
            print("Error: 'mainn_window.ui' not found. Please check the file path.")
            # --- MODIFIED: Fallback window now includes BOTH buttons ---
            self.setWindowTitle("Main Window (UI File Missing)")
            
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            
            self.btnQuickPredict = QtWidgets.QPushButton("Open Predict Window", self)
            self.btn_nav_predict = QtWidgets.QPushButton("Open Predict Window", self)
            self.btnQuickVisual = QtWidgets.QPushButton("Open Visualize Window", self) # Create new button
            self.btn_nav_visual = QtWidgets.QPushButton("Open Visualize Window", self) 
            
            
            layout.addWidget(self.btnQuickPredict)
            layout.addWidget(self.btn_nav_predict)
            layout.addWidget(self.btnQuickVisual) # Add new button
            layout.addWidget(self.btn_nav_visual) 
            self.setCentralWidget(central_widget)
            
            self.btnQuickPredict.clicked.connect(self.open_predict_window)
            self.btn_nav_predict.clicked.connect(self.open_predict_window)
            self.btnQuickVisual.clicked.connect(self.open_visualize_window) # Connect new button
            self.btn_nav_visual.clicked.connect(self.open_visualize_window) # Connect new button
            # --- END MODIFICATION ---
            
        except Exception as e:
            print(f"An error occurred loading mainn_window.ui: {e}")

    def open_predict_window(self):
        # Store as attribute so it doesn't get garbage-collected
        self.predict_window = PredictWindow()
        self.predict_window.show()

    # --- ADDED: Function to open the visualize window ---
    def open_visualize_window(self):
        # Store as attribute so it doesn't get garbage-collected
        self.visualize_window = VizWindow()
        self.visualize_window.show()
    # --- END ADDED FUNCTION ---

# -----------------------------------------------------------------
#  RUN THE APPLICATION
# -----------------------------------------------------------------

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow() # --- MODIFIED: Start the main app ---
    window.show()
    sys.exit(app.exec_())
    
    
    
    
    
    
    
    