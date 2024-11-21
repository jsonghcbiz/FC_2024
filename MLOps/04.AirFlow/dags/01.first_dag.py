from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

def hello():
    print('Hello World')

with DAG(
    'first_dags',
    schedule_interval="0 0 * * *",  # 분, 시, 일, 월, 요일  => 매일 자정 실행 
    start_date=datetime(2024, 11, 22),  # 시작 날짜 
    description='Simple DAG'
) as dag:
    #Task1 python operator 
    task1 = PythonOperator(
        task_id='print_hello_world',
        python_callable=hello, 
        dag=dag
    )

    #Task2는 보통 dummy operator 사용 
    task2 = DummyOperator(
        task_id='dummy_task',
        dag=dag
    )

    # 실행 순서 정의 
    task1 >> task2

