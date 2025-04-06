import os
from dotenv import load_dotenv
from WorkPDB import WorkPDB

load_dotenv()

# Инициализация подключения к БД
db_host = os.getenv("DB_HOST")
db_port = int(os.getenv("DB_PORT"))
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

# Создание экземпляра для работы с БД
db = WorkPDB(
    port=db_port,
    user=db_user,
    host=db_host,
    password=db_password
)