import numpy as np
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMessageBox

class PredictWindow(QtWidgets.QMainWindow):
    def __init__(self, trained_model=None, parent=None):
        super(PredictWindow, self).__init__(parent)
        
        uic.loadUi(r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_1\QT5\UI\predict_price.ui", self)

        # Store trained model from previous page
        self.model = trained_model

        # Connect button
        self.btn_predict.clicked.connect(self.predict_price)
        self.btn_back.clicked.connect(self.go_back)
        

        # Placeholder for result
        self.label_price.setText("Predicted Price: —")

    def predict_price(self):
        """Predict house price from user input."""
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please train a model first!")
            return

        try:
            rooms = float(self.line_rooms.text())
            poverty = float(self.line_poverty.text())
            ratio = float(self.line_ratio.text())

            # Prepare data (reshape to match model input)
            features = np.array([[rooms, poverty, ratio]])

            # Predict
            price = self.model.predict(features)[0]
            self.label_price.setText(f"Predicted Price: ${price:,.2f}")

        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{e}")

    def go_back(self):
        """Return to training window."""
        self.close()
        if self.parent():
            self.parent().show()
