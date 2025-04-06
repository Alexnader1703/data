from WorkPDB import WorkPDB
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()

db_host = os.getenv("DB_HOST")
db_port = int(os.getenv("DB_PORT"))
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

mydb = WorkPDB(
    port=db_port,
    user=db_user,
    host=db_host,
    password=db_password
)

print(mydb.info_db())

path=r"createdb.sql"
print(mydb.execute_sql_files(path) )

path_exel=r"UEFA Champions League 2016-2022 Data.xlsx"
print(mydb.load_excel_to_db(path_exel))
