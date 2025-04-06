import psycopg2
from psycopg2 import Error
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.mssql.information_schema import columns


class WorkPDB:
    def __init__(self, user, password, host, port):
        try:
            self.__user = user
            self.__password = password
            self.__host = host
            self.__port = port

            self.connection = psycopg2.connect(
                user=self.__user,
                password=self.__password,
                host=self.__host,
                port=self.__port
            )
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.connection.cursor()

            self.engine = create_engine(
                f'postgresql://{self.__user}:{self.__password}@{self.__host}:{self.__port}/uefadb')

        except (Exception, Error) as error:
            print(f"Ошибочка {error}")

    def info_db(self):
        try:
            info = "Информация о сервере\n"
            info += str(self.connection.get_dsn_parameters()) + "\n"

            self.cursor.execute("SELECT version();")
            record = self.cursor.fetchone()

            info += f"Вы подключены к - {record}\n"
            return info
        except (Exception, Error) as error:
            return str(error)

    def execute_sql(self,sql):
        try:
            self.cursor.execute(sql)
            results=self.cursor.fetchall()
            return results
        except(Exception,Error) as error:
            return error
        finally:
            self.connection.commit()
    def execute_sql_dict(self,sql):
        try:
            self.cursor.execute(sql)
            columns=[desc[0] for desc in self.cursor.description]
            results=self.cursor.fetchall()

            results_dict=[dict(zip(columns,row)) for row in columns]

            return results_dict

        except(Exception,Error) as error:
            return error
        finally:
            self.connection.commit()

    def execute_sql_files(self,filepath):
        try:
            with open(filepath, "r") as file:
                sql = file.read()
            sql_command = sql.split(";")

            for command in sql_command:
                self.cursor.execute(command)

            return self.cursor.fetchall()
        except(Exception,Error) as error:
            return error
        finally:
            self.connection.commit()

    def load_excel_to_db(self, excel_path):
        try:
            self.engine = create_engine(f'postgresql://{self.__user}:{self.__password}@{self.__host}:{self.__port}/uefadb')

            xls = pd.ExcelFile(excel_path)

            for sheet_name in xls.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                df.to_sql(sheet_name.lower(), self.engine, if_exists='replace', index=False)

            return "Все листы загружены"
        except Exception as error:
            return error

    def pd_return(self,sql):
        try:
            results = pd.read_sql_query(sql, con=self.engine)
        except(Exception,Error) as error:
            return error
        return results


