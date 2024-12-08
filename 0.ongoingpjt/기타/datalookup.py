import os
import pandas as pd

# current_path = os.getcwd()
# data_path = os.path.join(current_path, 'data', 'cgv_review_sentiment_01.csv')    # cgv_review_sentiment_01.csv 또는 megabox_review_sentiment.csv

# df = pd.read_csv(data_path)
# # df = pd.read_csv(data_path)
# print("컬럼 확인:", df.columns)  # Check the columns
# print("데이터 타입 확인:", df.dtypes)  # Check data types
# print(df['sentiment'].unique())
# # print("데이터 상위 몇 행 확인:", df.head())  # Display first few rows
# print(df.shape)



current_path = os.getcwd()
data_path = os.path.join(current_path, 'data', 'ratings.txt')    # cgv_review_sentiment_01.csv 또는 megabox_review_sentiment.csv

df = pd.read_csv(data_path, sep='\t')
# df = pd.read_csv(data_path)
print("컬럼 확인:", df.columns)  # Check the columns
print("데이터 타입 확인:", df.dtypes)  # Check data types
print(df['label'].unique())
# print("데이터 상위 몇 행 확인:", df.head())  # Display first few rows
print(df.shape)