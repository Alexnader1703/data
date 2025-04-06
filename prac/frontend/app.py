import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from dotenv import load_dotenv
from mpl_toolkits.mplot3d import Axes3D

# Загрузка конфигурации
load_dotenv()

# URL API-сервера
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Настройка страницы
st.set_page_config(
    page_title="Футбольная статистика",
    page_icon="⚽",
    layout="wide"
)


# Функция для получения данных с API
def fetch_data(endpoint):
    try:
        response = requests.get(f"{API_URL}/{endpoint}")
        response.raise_for_status()  # Вызовет исключение при HTTP-ошибке
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при получении данных: {e}")
        return None


# Заголовок приложения
st.title("⚽ Футбольная статистика")
st.markdown("Интерактивный анализ данных о футбольных матчах, игроках, стадионах и голах")

# Боковая панель с меню
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел",
    ["Главная", "Национальности и голы", "Стадионы и голы", "Позиции игроков",
     "Команды по странам", "3D анализ стадионов"]
)

# Главная страница
if page == "Главная":
    st.header("Добро пожаловать в приложение футбольной статистики!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("О приложении")
        st.write("""
        Это интерактивное приложение предоставляет статистические данные о футбольных матчах,
        игроках, голах и стадионах. Используйте меню слева для навигации по различным разделам.
        """)

    with col2:
        st.subheader("Доступная информация")
        st.markdown("""
        - **Национальности и голы**: Распределение голов по национальностям игроков
        - **Стадионы и голы**: Рейтинг стадионов по количеству забитых голов
        - **Позиции игроков**: Распределение игроков по позициям и национальностям
        - **Команды по странам**: Количество команд в разных странах
        - **3D анализ стадионов**: Взаимосвязь между вместимостью стадиона, посещаемостью и количеством голов
        """)

# Страница с анализом национальностей и голов
elif page == "Национальности и голы":
    st.header("Топ-10 стран по количеству голов")

    with st.spinner("Загрузка данных..."):
        data = fetch_data("nationality-goals")

        if data:
            df = pd.DataFrame(data)

            # Создаем интерактивную диаграмму с Plotly
            fig = px.bar(
                df,
                x='nationality',
                y='goals_count',
                title="Топ-10 стран по количеству голов",
                labels={"nationality": "Страна игрока", "goals_count": "Число голов"},
                color='goals_count',
                color_continuous_scale=px.colors.sequential.Viridis
            )

            st.plotly_chart(fig, use_container_width=True)

            # Показываем данные в таблице
            st.subheader("Данные:")
            st.dataframe(df)

# Страница стадионов и голов
elif page == "Стадионы и голы":
    st.header("Топ-10 стадионов по количеству голов")

    with st.spinner("Загрузка данных..."):
        data = fetch_data("stadium-goals")

        if data:
            df = pd.DataFrame(data)

            fig = px.bar(
                df,
                x='stadium_name',
                y='goals_count',
                title="Топ-10 стадионов по количеству голов",
                labels={"stadium_name": "Стадион", "goals_count": "Число голов"},
                color='goals_count',
                color_continuous_scale=px.colors.sequential.Magma
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Данные:")
            st.dataframe(df)

# Страница распределения игроков по позициям и национальностям
elif page == "Позиции игроков":
    st.header("Распределение игроков по позициям и топ-5 национальностям")

    with st.spinner("Загрузка данных..."):
        data = fetch_data("players-position-nationality")

        if data:
            df = pd.DataFrame(data)

            # Интерактивная группированная гистограмма
            fig = px.bar(
                df,
                x='position_clean',
                y='count',
                color='nationality',
                title="Распределение игроков по позициям и топ-5 национальностям",
                labels={"position_clean": "Позиция", "count": "Количество игроков", "nationality": "Национальность"},
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )

            st.plotly_chart(fig, use_container_width=True)

            # Фильтры для интерактивности
            st.subheader("Фильтры:")
            selected_positions = st.multiselect(
                "Выберите позиции для отображения:",
                options=sorted(df['position_clean'].unique()),
                default=sorted(df['position_clean'].unique())
            )

            selected_nationalities = st.multiselect(
                "Выберите национальности для отображения:",
                options=sorted(df['nationality'].unique()),
                default=sorted(df['nationality'].unique())
            )

            # Фильтрация данных
            filtered_df = df[
                (df['position_clean'].isin(selected_positions)) &
                (df['nationality'].isin(selected_nationalities))
                ]

            # Отображение отфильтрованных данных
            st.subheader("Отфильтрованные данные:")
            st.dataframe(filtered_df)

# Страница команд по странам
elif page == "Команды по странам":
    st.header("Количество различных команд по странам (Топ-10)")

    with st.spinner("Загрузка данных..."):
        data = fetch_data("teams-country")

        if data:
            df = pd.DataFrame(data)

            fig = px.bar(
                df,
                x='country',
                y='team_count',
                title="Количество различных команд по странам (Топ-10)",
                labels={"country": "Страна", "team_count": "Количество команд"},
                color='team_count',
                color_continuous_scale=px.colors.sequential.Blues
            )

            st.plotly_chart(fig, use_container_width=True)

            # Показываем карту мира с количеством команд
            st.subheader("Распределение команд по миру")
            fig_map = px.choropleth(
                df,
                locations="country",  # колонка с кодами стран
                locationmode="country names",  # режим определения местоположения по названию страны
                color="team_count",  # колонка для цветовой шкалы
                hover_name="country",  # данные для отображения при наведении
                title="Количество команд по странам",
                color_continuous_scale=px.colors.sequential.Plasma
            )

            st.plotly_chart(fig_map, use_container_width=True)

            st.subheader("Данные:")
            st.dataframe(df)

# Страница 3D анализа стадионов
elif page == "3D анализ стадионов":
    st.header("Зависимость количества голов от вместимости и посещаемости стадионов")

    with st.spinner("Загрузка данных..."):
        data = fetch_data("stadium-stats")

        if data:
            df = pd.DataFrame(data)

            # Создаем 3D график с plotly
            fig = px.scatter_3d(
                df,
                x='capacity',
                y='total_goals',
                z='average_attendance',
                color='country',
                hover_name='stadium_name',
                size='total_goals',  # размер точек пропорционален количеству голов
                opacity=0.7,
                labels={
                    "capacity": "Вместимость стадиона",
                    "total_goals": "Всего голов",
                    "average_attendance": "Средняя посещаемость",
                    "country": "Страна"
                },
                title="Взаимосвязь между вместимостью, посещаемостью и количеством голов"
            )

            # Настраиваем параметры отображения
            fig.update_layout(
                scene=dict(
                    xaxis_title='Вместимость стадиона',
                    yaxis_title='Всего голов',
                    zaxis_title='Средняя посещаемость'
                ),
                margin=dict(r=0, b=0, l=0, t=40)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Добавляем анализ корреляций
            st.subheader("Корреляционный анализ")

            # Вычисляем корреляцию между числовыми колонками
            corr = df[['capacity', 'total_goals', 'average_attendance']].corr()

            # Визуализируем корреляцию с помощью тепловой карты
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Матрица корреляций между параметрами стадионов"
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            # Показываем данные
            st.subheader("Данные стадионов:")
            st.dataframe(df)

# Подвал приложения
st.sidebar.markdown("---")
st.sidebar.info(
    "Это приложение разработано для анализа футбольной статистики. "
    "Используются данные о матчах, игроках, стадионах и голах."
)