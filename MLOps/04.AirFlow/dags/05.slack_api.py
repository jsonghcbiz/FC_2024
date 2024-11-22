from airflow import DAG
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.dates import days_ago
import os

os.environ['NO_PROXY'] = '*' # mac에서 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요

dag = DAG(
    dag_id="slack_test",
    start_date=days_ago(1),
    max_active_runs=1,
    catchup=False,
    schedule_interval="@once",
)

send_slack_message = SlackWebhookOperator(
    task_id="send_slack",
    slack_webhook_conn_id="slack_webhook", # 여기를 정의를 해주면 됩니다.
    message="Hello slack",
    dag=dag,
)

send_slack_message