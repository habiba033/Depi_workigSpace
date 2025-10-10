import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer, r2_score

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.predict_window import PredictWindow

class HousingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(HousingApp, self).__init__()
        uic.loadUi(r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_1\QT5\UI\Train_Model.ui", self)

        # Connect buttons
        self.btn_browse.clicked.connect(self.load_dataset)
        self.btn_train.clicked.connect(self.train_model)
        self.bttn_predict_page.clicked.connect(self.open_predict_page)
        self.btn_ok.clicked.connect(self.close)

        # Initialize attributes
        self.dataset = None
        self.model = None

        # Placeholder text
        self.line_dataset_path.setPlaceholderText("Select your dataset (e.g. housing.csv)")

    def load_dataset(self):
        """Let the user browse and select a dataset file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "", "CSV Files (*.csv)")
        if file_path:
            self.line_dataset_path.setText(file_path)
            try:
                self.dataset = pd.read_csv(file_path)
                self.label_status_train.setText(f"✅ Dataset loaded successfully! Shape: {self.dataset.shape}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read dataset:\n{e}")
        else:
            self.label_status_train.setText("⚠️ No file selected.")

    def train_model(self):
        """Train the Decision Tree Regressor using GridSearch."""
        if self.dataset is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        try:
            # --- Detect target column automatically ---
            target_col = None
            for col in self.dataset.columns:
                if col.strip().lower() in ['medv', 'price', 'target', 'value']:
                    target_col = col
                    break

            if not target_col:
                QMessageBox.critical(self, "Error", "Target column (like 'MEDV') not found in dataset.")
                return

            X = self.dataset.drop(target_col, axis=1)
            y = self.dataset[target_col]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Cross-validation
            cv_sets = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

            # Regressor + parameter grid
            regressor = DecisionTreeRegressor(random_state=0)
            params = {'max_depth': np.arange(1, 11)}

            # Scoring function
            scoring_fnc = make_scorer(r2_score)

            # Grid search
            grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_ 

            # Best model info
            best_depth = grid.best_params_.get('max_depth', None)
            best_score = grid.best_score_

            if best_depth is not None:
                text = f"✅ Model trained successfully!\nBest max_depth = {best_depth}\nCross-val R² = {best_score:.3f}"
            else:
                text = " Model trained but could not determine best depth."

            # Update label text
            self.label_status_train.setText(text)
            self.label_status_train.adjustSize()

        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"An error occurred:\n{e}")
    def open_predict_page(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Train the model first before predicting!")
            return
        self.predict_window = PredictWindow(trained_model=self.model, parent=self)
        self.predict_window.show()
        self.hide()
