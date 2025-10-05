# EDA with Interpretations (English)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
df = pd.read_csv("bike_data_preprocessed.csv")

# Ensure datetime parsing
df['start_time'] = pd.to_datetime(df['start_time'])
df['weekday'] = df['start_time'].dt.day_name()

# -------------------------------
# 1) Trips by Weekday
# -------------------------------
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='weekday', order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title("Number of Trips by Weekday")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n[Interpretation - Trips by Weekday]")
print("The chart shows the distribution of trips across the days of the week.")
print("Tuesday recorded the highest number of trips compared to other days.")
print("This suggests that users rely more on the service during mid-week, likely for commuting or work-related purposes.\n")

# -------------------------------
# 2) Trip Duration Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['trip_duration_minutes'], bins=50, kde=True)
plt.title("Trip Duration Distribution (minutes)")
plt.xlabel("Trip Duration (minutes)")
plt.ylabel("Count")
plt.xlim(0, 60)
plt.tight_layout()
plt.show()

print("\n[Interpretation - Trip Duration Distribution]")
print("The distribution indicates that most trips are short, concentrated under 20 minutes.")
print("This highlights the service as a quick commuting option rather than long-distance usage.")
print("There are a few outliers (long trips), but these are rare.\n")

# -------------------------------
# 3) Subscriber vs Customer Usage
# -------------------------------
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='user_type')
plt.title("Subscriber vs Customer Usage")
plt.tight_layout()
plt.show()

print("\n[Interpretation - Subscriber vs Customer Usage]")
print("The chart shows that Subscribers represent the majority of users compared to Customers.")
print("This suggests most people prefer subscription-based usage for regular commuting,")
print("while casual Customers are a smaller group using the service occasionally.\n")

# -------------------------------
# 4) Gender Breakdown by User Type
# -------------------------------
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='member_gender', hue='user_type')
plt.title("Gender Breakdown by User Type")
plt.tight_layout()
plt.show()

print("\n[Interpretation - Gender Breakdown by User Type]")
print("Male users dominate the service compared to female users.")
print("Most Subscribers are male, while females are relatively more represented among Customers.")
print("This indicates a potential gender gap in regular service adoption.\n")

# -------------------------------
# 5) Age Group Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='age_group', order=['Teen','20s','30s','40s','50+'])
plt.title("Trips by Age Group")
plt.tight_layout()
plt.show()

print("\n[Interpretation - Trips by Age Group]")
print("The chart shows that users in their 30s and 40s are the most active groups.")
print("Participation decreases significantly for users above 50 years old.")
print("This is reasonable since younger users are more likely to adopt biking for commuting or leisure.\n")