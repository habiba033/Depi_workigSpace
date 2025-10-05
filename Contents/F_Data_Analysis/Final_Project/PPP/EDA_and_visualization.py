# Part 2 - EDA (Exploratory Data Analysis)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read preprocessed data
df = pd.read_csv(r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\F_Data_Analysis\Final_Project\PPP\bike_data_preprocessed.csv")

# -------------------------------
# 1) Trips by Weekday
# -------------------------------
df['weekday'] = pd.to_datetime(df['start_time']).dt.day_name()

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='weekday', order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title("Number of Trips by Weekday")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 2) Trip Duration Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['trip_duration_minutes'], bins=50, kde=True)
plt.title("Trip Duration Distribution (minutes)")
plt.xlabel("Trip Duration (minutes)")
plt.ylabel("Count")
plt.xlim(0, 60)  # Focus on trips under 60 minutes
plt.tight_layout()
plt.show()

# -------------------------------
# 3) Subscriber vs Customer Usage
# -------------------------------
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='user_type')
plt.title("Subscriber vs Customer Usage")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.show()

# -------------------------------
# 4) Gender Breakdown by User Type
# -------------------------------
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='member_gender', hue='user_type')
plt.title("Gender Breakdown by User Type")
plt.tight_layout()
plt.show()

# -------------------------------
# 5) Age Group Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='age_group', order=['Teen','20s','30s','40s','50+'])
plt.title("Trips by Age Group")
plt.tight_layout()
plt.show()