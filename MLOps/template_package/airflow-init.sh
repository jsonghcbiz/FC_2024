#!/bin/bash

if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "Initializing Airflow DB and creating admin user..."
    airflow db init
    airflow users create \
        --username jesong \
        --password 0000 \
        --firstname jenn \
        --lastname song \
        --role Admin \
        --email jsong.hcbiz@gmail.com
else
    echo "Airflow DB already initialized. Skipping DB init."
fi