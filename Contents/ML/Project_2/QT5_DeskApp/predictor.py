
import sys
from PyQt5 import QtWidgets, uic, QtCore
import joblib
import numpy as np
import pandas as pd

class PredictWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(PredictWindow, self).__init__()
        
        try:
            uic.loadUi(r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_2\QT5_DeskApp\ui\predicttt_window.ui", self)
        except FileNotFoundError:
            print("Error: 'predicttt_window.ui' not found. Cannot load prediction window.")
            self.close()
            return
        except Exception as e:
            print(f"An error occurred loading predicttt_window.ui: {e}")
            self.close()
            return

        try:
            model_path = r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_2\income_predictor_model.pkl"
            scaler_path = r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_2\scaler.pkl"
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            print("Error: Model or scaler .pkl file not found. Please check paths.")
            self.predict_button.setEnabled(False)
            self.status_label.setText("Error: Model files not found.")
            return
        except Exception as e:
            print(f"An error occurred loading model/scaler: {e}")
            self.predict_button.setEnabled(False)
            self.status_label.setText("Error loading model.")
            return

        # ComboBox items
        self.education_items = [
            'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
            'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
            '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'
        ]
        self.workclass_items = [
            'Private','Self-emp-not-inc','Self-emp-inc','Federal-gov',
            'Local-gov','State-gov','Without-pay','Never-worked'
        ]

        if hasattr(self, 'education_input') and hasattr(self, 'workclass_input'):
            self.education_input.addItems(self.education_items)
            self.workclass_input.addItems(self.workclass_items)
        else:
            print("Error: UI elements 'education_input' or 'workclass_input' not found in .ui file.")

        self.education_map = {
            'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, 
            '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 
            'Assoc-acdm': 11, 'Assoc-voc': 12, 'Bachelors': 13, 'Masters': 14, 
            'Prof-school': 15, 'Doctorate': 16
        }

        if hasattr(self, 'predict_button'):
            self.predict_button.clicked.connect(self.make_prediction)
        else:
            print("Error: 'predict_button' not found in .ui file.")

        self.backToMain.clicked.connect(self.go_back_to_main)



    def go_back_to_main(self):
        self.close()  # Close the current window
    
    def make_prediction(self):
        print("age_input exists:", hasattr(self, 'age_input'))
        print("hours_input exists:", hasattr(self, 'hours_input'))
        print("capital_gain_input exists:", hasattr(self, 'capital_gain_input'))
        print("capital_loss_input exists:", hasattr(self, 'capital_loss_input'))

        if not hasattr(self, 'status_label'):
            print("Status label not found")
            return
        self.status_label.setText("")
        
        try:
            age_text = self.age_input.text().strip()
            hours_text = self.hours_input.text().strip()
            gain_text = self.capital_gain_input.text().strip()
            loss_text = self.capital_loss_input.text().strip()

            if not all([age_text, hours_text, gain_text, loss_text]):
                self.status_label.setText("Error: All number fields must be filled.")
                self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 10px")
                return

            try:
                age = float(age_text)
                hours = float(hours_text)
                gain = float(gain_text)
                loss = float(loss_text)
            except ValueError:
                self.status_label.setText("Error: Please enter valid numbers.")
                self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 10px")
                return

            
            edu = self.education_input.currentText()
            work = self.workclass_input.currentText()

            input_data = {
                'age': age,
                'education-num': self.education_map[edu],
                'capital-gain': np.log(gain + 1),
                'capital-loss': np.log(loss + 1),
                'hours-per-week': hours,
                'workclass_Federal-gov': 1 if work=='Federal-gov' else 0,
                'workclass_Local-gov': 1 if work=='Local-gov' else 0,
                'workclass_Private': 1 if work=='Private' else 0,
                'workclass_Self-emp-inc': 1 if work=='Self-emp-inc' else 0,
                'workclass_Self-emp-not-inc': 1 if work=='Self-emp-not-inc' else 0,
                'workclass_State-gov': 1 if work=='State-gov' else 0,
                'workclass_Without-pay': 1 if work=='Without-pay' else 0,
                'workclass_Never-worked': 1 if work=='Never-worked' else 0
            }

            df = pd.DataFrame([input_data])
            for col in self.model.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            
            # ... (code before this) ...
            df = df[self.model.feature_names_in_]
            
            # Use the EXACT same order as in model.py
            num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'] # <--- THIS IS THE FIX
            
            # This logic can be simplified and made safer
            if all(col in df.columns for col in num_cols):
                df[num_cols] = self.scaler.transform(df[num_cols])
            else:
                print("Error: Not all numerical columns were found in the DataFrame before scaling.")
                self.status_label.setText("Error: Scaling columns missing.")
                return

            df = df.reindex(columns=self.model.feature_names_in_, fill_value=0)
            # ... (code after this) ...
            pred = self.model.predict(df)[0]

            if hasattr(self, 'result_label'):
                result_text = "💰 Income > 50K" if pred == 1 else "💼 Income ≤ 50K"
                wrapped_text = f"<p style='text-align:center;'>{result_text.replace(' ', '<br>')}</p>"
                self.result_label.setText(wrapped_text)
                self.result_label.setStyleSheet("font-weight: bold; font-size: 30px")
                self.result_label.setAlignment(QtCore.Qt.AlignCenter)
                self.result_label.setWordWrap(True)
            else:
                print(f"Prediction: {pred}")

        # except ValueError:
        #     self.status_label.setText("Error: Please enter valid numbers.")
        #     self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 10px")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px")
            print(f"An error occurred during prediction: {e}")



