from airflow import DAG
from datetime import datetime
from airflow.operators.bash_operator import BashOperator

args = {
    'start_date': datetime(2024, 11, 22),
    'retries': 1,  # 실패 시 재시도 횟수 
    # 'retry_delay': datetime.timedelta(minutes=2)
}

dag = DAG(
    'second_dag',
    default_args=args,
    # schedule_interval="0 0 * * *",  # 분, 시, 일, 월, 요일  => 매일 자정 실행 
    # start_date=datetime(2024, 11, 22),  # 시작 날짜 
    description='Simple DAG - bash command'
)

task1 = BashOperator(
    task_id='print_date',
    bash_command='date', # 현재 날짜 출력 
    dag=dag
)

task2 = BashOperator(
    task_id='print_sleep',
    bash_command='sleep 5',  # 5초 대기 
    dag=dag
)

task3 = BashOperator(
    task_id='print_hello',
    bash_command='echo "Hello World"', # 문자열 출력 
    dag=dag
)


# 의존성 설정 (실행 순서)
task1 >> task2 >> task3

