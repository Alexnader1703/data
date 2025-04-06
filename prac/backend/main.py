from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
from database import db
import models

app = FastAPI(title="Футбольная статистика API")

# Настройка CORS для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "API футбольной статистики"}


@app.get("/nationality-goals", response_model=List[models.GoalsByNationality])
def get_nationality_goals():
    sql_nationality_goals = """
        SELECT p."NATIONALITY" AS nationality, COUNT(g."GOAL_ID") AS goals_count
        FROM goals g
        JOIN players p ON g."PID" = p."PLAYER_ID"
        GROUP BY p."NATIONALITY"
        ORDER BY goals_count DESC
        LIMIT 10;
        """
    try:
        df = db.pd_return(sql_nationality_goals)
        # Преобразуем DataFrame в список словарей для совместимости с Pydantic
        result = df.to_dict(orient="records")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {str(e)}")


@app.get("/stadium-goals", response_model=List[models.GoalsByStadium])
def get_stadium_goals():
    sql_stadium_goals = """
        SELECT s."NAME" AS stadium_name, COUNT(g."GOAL_ID") AS goals_count
        FROM goals g
        JOIN matches m ON g."MATCH_ID" = m."MATCH_ID"
        JOIN stadiums s ON m."STADIUM" = s."NAME"
        GROUP BY s."NAME"
        ORDER BY goals_count DESC
        LIMIT 10;
        """
    try:
        df = db.pd_return(sql_stadium_goals)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {str(e)}")


@app.get("/players-position-nationality", response_model=List[models.PlayerPositionNationality])
def get_players_position_nationality():
    sql_players_position_nationality = """
    SELECT p."POSITION" AS position, p."NATIONALITY" AS nationality
    FROM players p;
    """
    try:
        df = db.pd_return(sql_players_position_nationality)
        df = df.dropna(subset=['position', 'nationality'])

        standard_positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
        df['position_clean'] = df['position'].apply(
            lambda x: x if x in standard_positions else 'Other'
        )

        top_nationalities = df['nationality'].value_counts().nlargest(5).index
        df_top = df[df['nationality'].isin(top_nationalities)]

        # Агрегируем данные для получения количества
        result_df = df_top.groupby(['position_clean', 'nationality']).size().reset_index(name='count')

        return result_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {str(e)}")


@app.get("/teams-country", response_model=List[models.TeamsByCountry])
def get_teams_country():
    sql_teams_country = """
    SELECT "COUNTRY" AS country, COUNT(*) AS team_count
    FROM teams
    GROUP BY "COUNTRY"
    ORDER BY team_count DESC
    LIMIT 10;
    """
    try:
        df = db.pd_return(sql_teams_country)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {str(e)}")


@app.get("/stadium-stats", response_model=List[models.StadiumStats])
def get_stadium_stats():
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
    try:
        df = db.pd_return(sql_query)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {str(e)}")