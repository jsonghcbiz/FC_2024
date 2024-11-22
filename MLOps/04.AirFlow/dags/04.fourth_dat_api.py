from airflow import DAG
import requests
import logging
import json
from datetime import datetime
from airflow.operators.python import PythonOperator
import os
# API 요청 -> 데이터 수집 -> 작업

# # mac에서 contaier를 띄운 다음 airflow에서 외부로 fetch 요청을 할 때 발생하는 이슈 => log가 찍히지 않고 계속 runnning 상태
os.environ["NO_PROXY"] = "*"   # log가 찍히지 않는 문제 해결

default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 9, 22),
    'retries': 1,
}

API_KEY = 'e8833daf3081187c1de5174a1e113363'

# 1. 데이터 수집 함수
def fetch_weather_data(**context): 
    logging.info("Fetching weather data from OpenWeather API...")
    REQUEST_URL = f'https://api.openweathermap.org/data/2.5/weather?lat=37.5172&lon=127.0473&appid={API_KEY}'
    try: 
        response = requests.get(REQUEST_URL, timeout=5)
        response.raise_for_status() # 응답 코드가 200이 아닌 경우 예외 발생
        logging.info(f"Weather data fetched successfully: {response.json()}")
        context['ti'].xcom_push(key='weather_data', value=response.json())
    except requests.exceptions.Timeout:
        logging.error("Request timed out")
        raise  # 타임아웃 발생 시 태스크를 실패로 표시
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        raise  # 다른 요청 예외 발생 시 태스크 실패

# 2. 데이터 처리 함수
def preprocess_weather_data(**context): 
    logging.info("Processing weather data...")
    raw_data = context['ti'].xcom_pull(
        key='weather_data', 
        task_ids='fetch_weather_data'  # task_ids 명시
    )
    if raw_data:
        processed_data = {
            "location": raw_data.get("name"),
            "temperature": raw_data["main"].get("temp"),
            "humidity": raw_data["main"].get("humidity"),
            "pressure": raw_data["main"].get("pressure"),
            "weather": raw_data["weather"][0].get("description"),
            "wind_speed": raw_data["wind"].get("speed"),
            "timestamp": raw_data.get("dt")
        }
        logging.info(f"Processed weather data: {processed_data}")
        context['ti'].xcom_push(key='processed_weather_data', value=processed_data)
    else:
        logging.error("No data to process")
    

def save_data(**context): 
    logging.info("Saving processed weather data to /tmp/weather_data.json...")
    processed_data = context['ti'].xcom_pull(key='processed_weather_data', task_ids='process_weather_data')  # task_ids 명시
    
    # 데이터 저장
    if processed_data:
        output_path = '/tmp/weather_data.json'  # 절대 경로로 변경
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Weather data saved successfully at {output_path}")
    else:
        logging.error("No processed data to save")

# def dag 함수로 묶을 수 있을까?
dag = DAG(
    'weather_data_pipeline',
    default_args=default_args,
    description='Weather data collection pipeline DAG', 
    start_date=datetime(2024, 11, 21),
    schedule_interval='@daily',
    catchup=False # 과거 실행 주기 무시
)
    
# Move these operator definitions inside the function and fix indentation
fetch_data_task = PythonOperator(
    task_id='fetch_weather_data',
    python_callable=fetch_weather_data,
    provide_context=True,
    dag=dag
    )
preprocess_task2 = PythonOperator(
    task_id='preprocess_weather_data',
    python_callable=preprocess_weather_data,
    provide_context=True,
    dag=dag
    )
save_data_task3 = PythonOperator(
    task_id='save_weather_data',
    python_callable=save_data,
    provide_context=True,
    dag=dag
    )

# task 의존성 설정
fetch_data_task >> preprocess_task2 >> save_data_task3
    
# return dag  # def 함수 활용 시, return statement 추가
# dag = weather_data_pipeline() # 함수 호출 시, 변수 할당 필요