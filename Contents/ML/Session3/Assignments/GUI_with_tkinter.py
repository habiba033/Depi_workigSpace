'''
GUI with Tkinter => Predict Salary based on Years of Experience
'''
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ======== Load and prepare the data ========
data = pd.read_csv(r'Contents/ML/Session3/Assignments/Salary_Data.csv')

x = data.iloc[:, :-1]   
y = data.iloc[:, 1]     

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

# ======== GUI ==========
root = tk.Tk()
root.title("Salary Prediction")
root.geometry("400x200")
root.config(bg="lightblue") 

label = tk.Label(root, text="Enter Years of Experience:" ,bg="lightblue", fg="black")
label.pack(pady=10)

entry = tk.Entry(root)
entry.pack(pady=5)

def predict_salary():
    user_input = entry.get()   

    if not user_input:
        messagebox.showerror("Error", "Please enter a number!")
        return

    try:
        years = float(user_input)
    except:
        messagebox.showerror("Error", "Only numbers are allowed!")
        return

    if years < 0 or years > 65:
        messagebox.showerror("Error", "Please enter a value between 0 and 65.")
        return

    salary = model.predict([[years]])[0]
    messagebox.showinfo("Prediction", f"Predicted Salary: ${salary:,.2f}")

button = tk.Button(root, text="Predict Salary", command=predict_salary,bg="green", fg="white")
button.pack(pady=20)

root.mainloop()