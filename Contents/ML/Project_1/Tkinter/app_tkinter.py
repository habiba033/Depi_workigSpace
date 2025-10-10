import tkinter as tk
from tkinter import filedialog, messagebox, font
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

try:
    from utils.visuals import ModelComplexity
except ImportError:
    messagebox.showerror(
        "Missing File",
        "Error: The 'visuals.py' file was not found.\n\nPlease make sure it is in the same directory as this script."
    )
    exit()

class HousingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ Housing Price Predictor ✨")
        self.root.geometry("650x750")
        self.root.resizable(False, False)

        # --- Style Configuration (Kawaii Purple Theme) ---
        self.BG_COLOR = "#F5F3FF"
        self.CONTAINER_BG = "#FFFFFF"
        self.PRIMARY_COLOR = "#7E57C2"
        self.PRIMARY_HOVER = "#673AB7"
        self.SECONDARY_COLOR = "#9575CD"
        self.SECONDARY_HOVER = "#8E6DD2"
        self.TEXT_COLOR = "#4C3B71"
        self.SECONDARY_TEXT = "#59516d"
        self.BORDER_COLOR = "#D9D2E9"
        self.SUCCESS_COLOR = "#34A853"
        self.ERROR_COLOR = "#EA4335"
        
        self.FONT_FAMILY = "Segoe UI"
        self.FONT_BOLD = font.Font(family=self.FONT_FAMILY, size=10, weight="bold")
        self.FONT_NORMAL = font.Font(family=self.FONT_FAMILY, size=10)
        self.FONT_TITLE = font.Font(family=self.FONT_FAMILY, size=24, weight="bold")
        self.FONT_RESULT = font.Font(family=self.FONT_FAMILY, size=16, weight="bold")

        self.root.config(bg=self.BG_COLOR, padx=40, pady=30)

        # --- Application State ---
        self.dataset = None
        self.model = None
        self.target_col = None
        # Assuming these features based on standard housing datasets
        self.feature_cols = ['RM', 'LSTAT', 'PTRATIO']

        # --- Initialize UI Elements ---
        self.create_widgets()

    def create_widgets(self):
        """Creates and styles all the widgets for the application."""
        tk.Label(
            self.root, text="Predict House Price", font=self.FONT_TITLE, 
            bg=self.BG_COLOR, fg=self.TEXT_COLOR
        ).pack(pady=(0, 25))

        # --- 1. Dataset Section ---
        dataset_frame = self.create_rounded_frame()
        dataset_frame.pack(fill="x", pady=10)
        self.create_styled_button(dataset_frame, "📂 Load Dataset", self.load_dataset).pack(pady=10)
        self.label_status = tk.Label(dataset_frame, text="Status: No dataset loaded.", fg=self.SECONDARY_TEXT, bg=self.CONTAINER_BG, font=self.FONT_NORMAL)
        self.label_status.pack(pady=(0, 10))

        # --- 2. Training & Visualization Section ---
        train_frame = self.create_rounded_frame()
        train_frame.pack(fill="x", pady=10)
        self.create_styled_button(train_frame, "🚀 Train Model", self.train_model).pack(pady=10)
        self.create_styled_button(
            train_frame, "📈 Show Model Complexity", self.show_complexity, 
            color=self.SECONDARY_COLOR, hover_color=self.SECONDARY_HOVER
        ).pack(pady=10)

        # --- 3. Prediction Section ---
        predict_frame = self.create_rounded_frame()
        predict_frame.pack(fill="x", pady=10)
        self.entry_rooms = self.create_labeled_entry(predict_frame, f"{self.feature_cols[0]} (Total rooms):")
        self.entry_poverty = self.create_labeled_entry(predict_frame, f"{self.feature_cols[1]} (Poverty level %):")
        self.entry_ratio = self.create_labeled_entry(predict_frame, f"{self.feature_cols[2]} (Student–teacher ratio):")
        self.create_styled_button(predict_frame, "💰 Predict Price", self.predict_price).pack(pady=20)
        self.label_result = tk.Label(predict_frame, text="Predicted Price: $...", font=self.FONT_RESULT, fg=self.TEXT_COLOR, bg=self.CONTAINER_BG)
        self.label_result.pack(pady=(0, 15))

    # --- UI Helper Methods ---
    def create_rounded_frame(self):
        return tk.Frame(self.root, bg=self.CONTAINER_BG, bd=1, relief="solid", highlightbackground=self.BORDER_COLOR, highlightthickness=1)

    def create_styled_button(self, parent, text, command, color=None, hover_color=None):
        bg_color = color or self.PRIMARY_COLOR
        hover_bg_color = hover_color or self.PRIMARY_HOVER
        button = tk.Button(parent, text=text, command=command, font=self.FONT_BOLD, bg=bg_color, fg="white", relief="flat", pady=8, padx=15, activebackground=hover_bg_color, activeforeground="white", cursor="hand2")
        button.bind("<Enter>", lambda e, c=hover_bg_color: e.widget.config(bg=c))
        button.bind("<Leave>", lambda e, c=bg_color: e.widget.config(bg=c))
        return button

    def create_labeled_entry(self, parent, label_text):
        frame = tk.Frame(parent, bg=self.CONTAINER_BG)
        frame.pack(pady=8, padx=20, fill="x")
        tk.Label(frame, text=label_text, font=self.FONT_NORMAL, bg=self.CONTAINER_BG, fg=self.TEXT_COLOR, width=25, anchor="w").pack(side="left")
        entry = tk.Entry(frame, font=self.FONT_NORMAL, relief="solid", bd=1, highlightthickness=1, highlightcolor=self.PRIMARY_COLOR, highlightbackground=self.BORDER_COLOR, width=20)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    # --- Backend Logic ---
    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path: return
        try:
            data = pd.read_csv(file_path)
            self.target_col = next((c for c in data.columns if c.lower() in ['medv', 'price']), None)
            
            if self.target_col and all(f in data.columns for f in self.feature_cols):
                self.dataset = data
                self.label_status.config(text=f"✅ Loaded: {self.dataset.shape[0]} rows", fg=self.SUCCESS_COLOR)
                self.model = None # Reset model when new data is loaded
            else:
                messagebox.showerror("Invalid Dataset", f"Dataset must contain target column ('MEDV' or 'price') and feature columns: {', '.join(self.feature_cols)}")
                self.label_status.config(text="Status: Invalid dataset.", fg=self.ERROR_COLOR)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")

    def train_model(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        try:
            X = self.dataset[self.feature_cols]
            y = self.dataset[self.target_col]
            self.model = DecisionTreeRegressor(max_depth=4).fit(X, y) # A good depth from the complexity plot
            messagebox.showinfo("Success", f"✅ Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during training:\n{e}")

    def show_complexity(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        try:
            X = self.dataset.drop(self.target_col, axis=1) # Use all other columns for complexity analysis
            y = self.dataset[self.target_col]
            # This calls the function from your visuals.py file to show the plot
            ModelComplexity(X, y)
        except Exception as e:
            messagebox.showerror("Plotting Error", f"Could not generate complexity plot:\n{e}")

    def predict_price(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return
        try:
            inputs = [float(self.entry_rooms.get()), float(self.entry_poverty.get()), float(self.entry_ratio.get())]
            price = self.model.predict([inputs])[0]
            self.label_result.config(text=f"Predicted Price: ${price:,.2f}", fg=self.PRIMARY_COLOR)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers in all prediction fields.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HousingApp(root)
    root.mainloop()