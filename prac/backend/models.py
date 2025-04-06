from pydantic import BaseModel
from typing import List, Optional

class GoalsByNationality(BaseModel):
    nationality: str
    goals_count: int

class GoalsByStadium(BaseModel):
    stadium_name: str
    goals_count: int

class PlayerPositionNationality(BaseModel):
    position_clean: str
    nationality: str
    count: int

class TeamsByCountry(BaseModel):
    country: str
    team_count: int

class StadiumStats(BaseModel):
    stadium_name: str
    capacity: int
    country: str
    total_goals: int
    average_attendance: float