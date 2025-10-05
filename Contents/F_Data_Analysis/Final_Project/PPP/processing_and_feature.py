# Part 2 - Preprocessing & Feature Engineering

import pandas as pd
import numpy as np

# 1) Load cleaned data
df = pd.read_csv(r"Contents\F_Data_Analysis\Final_Project\PPP\bike_data_preprocessed.csv")
print("Shape before:", df.shape)

# # Preview
# print(df.head())

# # -------------------------------
# # 2) Handle Missing Values
# # -------------------------------
# # Example: fill gender 'Other' for missing, drop NA if too many
# df['member_gender'] = df['member_gender'].fillna('Other')
# df['member_birth_year'] = df['member_birth_year'].fillna(df['member_birth_year'].median())
# df['user_type'] = df['user_type'].fillna('Unknown')

# # Drop rows with missing start/end station (can't analyze trips without them)
# df = df.dropna(subset=['start_station_id','end_station_id'])

# # -------------------------------
# # 3) Feature Engineering
# # -------------------------------

# # Convert start_time, end_time to datetime
# df['start_time'] = pd.to_datetime(df['start_time'])
# df['end_time']   = pd.to_datetime(df['end_time'])

# # Trip duration in minutes
# df['trip_duration_minutes'] = df['duration_sec'] / 60

# # Weekend flag (0=weekday, 1=weekend)
# df['weekend_flag'] = df['start_time'].dt.dayofweek.apply(lambda x: 1 if x>=5 else 0)

# # Age (if not in dataset already)
# current_year = pd.Timestamp.today().year
# df['age'] = current_year - df['member_birth_year']

# # Age groups
# def age_group(age):
#     if age < 20: return 'Teen'
#     elif 20 <= age < 30: return '20s'
#     elif 30 <= age < 40: return '30s'
#     elif 40 <= age < 50: return '40s'
#     else: return '50+'

# df['age_group'] = df['age'].apply(age_group)

# # -------------------------------
# # 4) Encoding (if needed later)
# # -------------------------------
# # Example: Convert categorical to numeric
# from sklearn.preprocessing import LabelEncoder

# le_gender = LabelEncoder()
# df['gender_encoded'] = le_gender.fit_transform(df['member_gender'])

# le_usertype = LabelEncoder()
# df['user_type_encoded'] = le_usertype.fit_transform(df['user_type'])

# # # -------------------------------
# # # Final check
# # # -------------------------------
# # print("Shape after preprocessing:", df.shape)
# # print(df[['trip_duration_minutes','weekend_flag','age','age_group']].head())

# # # Save preprocessed data
# # df.to_csv("bike_data_preprocessed.csv", index=False)
# # print("Preprocessed file saved: bike_data_preprocessed.csv")


null = df.isnull().sum()

ratio = round(((null/df.shape[0])*100),2).astype(str)+"%"
# null / number of rows 
print(pd.DataFrame({ "Num_Nulls":null , "Ratio %" : ratio}).T)