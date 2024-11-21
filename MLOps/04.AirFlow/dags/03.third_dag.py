from airflow import DAG
from datetime import datetime
from airflow.operators.bash_operator import BashOperator

args = {
    'start_date': datetime(2024, 11, 21),
    'retries': 1,  # 실패 시 재시도 횟수 
    # 'retry_delay': datetime.timedelta(minutes=2)
}

dag = DAG(
    'third_dag',
    default_args=args,
    description='Simple DAG - bash command'
)

task1 = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag
)

task2 = BashOperator(
    task_id='print_sleep',
    bash_command='echo "Hello, World - Task2"',
    dag=dag
)

task3 = BashOperator(
    task_id='print_hello',
    bash_command='echo "Hello, World - Task3"',
    dag=dag
)


# 의존성 설정 (실행 순서)
task1 >> [task2, task3]

