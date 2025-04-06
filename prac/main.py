
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from WorkPDB import WorkPDB
from mpl_toolkits.mplot3d import Axes3D
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


def nationality_goal():
    sql_nationality_goals = """
        SELECT p."NATIONALITY" AS nationality, COUNT(g."GOAL_ID") AS goals_count
        FROM goals g
        JOIN players p ON g."PID" = p."PLAYER_ID"
        GROUP BY p."NATIONALITY"
        ORDER BY goals_count DESC
        LIMIT 10;
        """

    df_nationality_goals = mydb.pd_return(sql_nationality_goals)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_nationality_goals, x='nationality', y='goals_count', palette='viridis')
    plt.title("Топ-10 стран по количеству голов")
    plt.xlabel("Страна игрока")
    plt.ylabel("Число голов")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def stadium_goals():
    sql_stadium_goals = """
        SELECT s."NAME" AS stadium_name, COUNT(g."GOAL_ID") AS goals_count
        FROM goals g
        JOIN matches m ON g."MATCH_ID" = m."MATCH_ID"
        JOIN stadiums s ON m."STADIUM" = s."NAME"
        GROUP BY s."NAME"
        ORDER BY goals_count DESC
        LIMIT 10;
        """

    df_stadium_goals = mydb.pd_return(sql_stadium_goals)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_stadium_goals, x='stadium_name', y='goals_count', palette='magma')
    plt.title("Топ-10 стадионов по количеству голов")
    plt.xlabel("Стадион")
    plt.ylabel("Число голов")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def players_position_nationality():
    sql_players_position_nationality = """
    SELECT p."POSITION" AS position, p."NATIONALITY" AS nationality
    FROM players p;
    """
    df_players_position_nationality = mydb.pd_return(sql_players_position_nationality)

    df_players_position_nationality = df_players_position_nationality.dropna(subset=['position', 'nationality'])


    standard_positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']

    df_players_position_nationality['position_clean'] = df_players_position_nationality['position'].apply(
        lambda x: x if x in standard_positions else 'Other'
    )

    top_nationalities = df_players_position_nationality['nationality'].value_counts().nlargest(5).index
    df_players_top = df_players_position_nationality[
        df_players_position_nationality['nationality'].isin(top_nationalities)]

    plt.figure(figsize=(14, 8))
    sns.countplot(data=df_players_top, x='position_clean', hue='nationality', palette='Set2')
    plt.title("Распределение игроков по позициям и топ-5 национальностям")
    plt.xlabel("Позиция")
    plt.ylabel("Количество игроков")
    plt.legend(title='Национальность', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def teams_country():
    sql_teams_country = """
    SELECT "COUNTRY" AS country, COUNT(*) AS team_count
    FROM teams
    GROUP BY "COUNTRY"
    ORDER BY team_count DESC
    LIMIT 10;
    """

    df_teams_country = mydb.pd_return(sql_teams_country)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_teams_country, x='country', y='team_count', palette='Blues_d')
    plt.title("Количество различных команд по странам (Топ-10)")
    plt.xlabel("Страна")
    plt.ylabel("Количество команд")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def goals_capacity_attendance_3d():
    sql_query = """
    SELECT 
        s."NAME" AS stadium_name,
        s."CAPACITY" AS capacity,
        s."COUNTRY" AS country,
        COUNT(g."GOAL_ID") AS total_goals,
        AVG(m."ATTENDANCE") AS average_attendance
    FROM stadiums s
    JOIN matches m ON s."NAME" = m."STADIUM"
    JOIN goals g ON m."MATCH_ID" = g."MATCH_ID"
    GROUP BY s."NAME", s."CAPACITY", s."COUNTRY"
    HAVING COUNT(g."GOAL_ID") > 0
    ORDER BY total_goals DESC;
    """

    df_stadiums = mydb.pd_return(sql_query)

    # Преобразуем страну в числовые коды для раскраски
    df_stadiums['country_factor'], country_labels = df_stadiums['country'].factorize()

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')

    sc = ax.scatter(
        df_stadiums['capacity'],
        df_stadiums['total_goals'],
        df_stadiums['average_attendance'],
        c=df_stadiums['country_factor'],
        cmap='viridis',
        alpha=0.8
    )

    ax.set_xlabel("Вместимость стадиона")
    ax.set_ylabel("Всего голов")
    ax.set_zlabel("Средняя посещаемость")
    plt.title("Зависимость количества голов от вместимости и посещаемости")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Страна (цвет)")

    plt.show()



nationality_goal()
stadium_goals()
players_position_nationality()
teams_country()
goals_capacity_attendance_3d()