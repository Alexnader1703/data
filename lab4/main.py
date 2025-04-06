import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import sys
from  lin_regres import MyLineReg
# Импортируем вашу модель
# Предполагаю, что код MyLineReg находится в этом же файле
# Если она в отдельном файле, нужно импортировать иначе

# 1. Загрузка набора данных
print("Загрузка данных...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
pd.set_option('display.max_columns', None)
print("Все столбцы в train_data:")
print(train_data.columns.tolist())
# 2. Разведочный анализ данных
print("\n--- РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ ---")
# a. Количество строк и столбцов
print(f"Количество строк в датафрейме: {train_data.shape[0]}")
print(f"Количество столбцов в датафрейме: {train_data.shape[1]}")

# b. Размер датафрейма в памяти
memory_usage = train_data.memory_usage(deep=True).sum()
print(f"Датафрейм занимает в памяти: {memory_usage / (1024 * 1024):.2f} MB")

# c. Статистика для интервальных переменных
print("\nСтатистика для интервальных переменных:")
numeric_columns = train_data.select_dtypes(include=['number']).columns
numeric_stats = train_data[numeric_columns].describe(percentiles=[.25, .5, .75])
print(numeric_stats)

# d. Статистика для категориальных переменных
print("\nСтатистика для категориальных переменных:")
categorical_columns = train_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    mode_value = train_data[col].mode()[0]
    mode_count = train_data[col].value_counts()[mode_value]
    print(f"Колонка '{col}':")
    print(f"  Мода: {mode_value}")
    print(f"  Количество вхождений моды: {mode_count}")
    print(f"  Распределение значений:")
    print(train_data[col].value_counts().head())
    print()

# 3. Подготовка датасета к построению моделей ML
print("\n--- ПОДГОТОВКА ДАННЫХ ---")

# a. Анализ и обработка пропусков
print("\nАнализ пропусков:")
missing_values = train_data.isnull().sum()
missing_percent = (missing_values / len(train_data)) * 100
missing_df = pd.DataFrame({'Количество пропусков': missing_values,
                           'Процент пропусков': missing_percent})
print(missing_df[missing_df['Количество пропусков'] > 0])

# Обработка пропусков (заполнение или удаление)
if missing_values.sum() > 0:
    print("\nОбработка пропусков:")
    # Для числовых колонок заполним медианой
    for col in train_data.select_dtypes(include=['number']).columns:
        if train_data[col].isnull().sum() > 0:
            median_value = train_data[col].median()
            train_data[col].fillna(median_value, inplace=True)
            test_data[col].fillna(median_value, inplace=True)
            print(f"  Заполнили пропуски в '{col}' медианным значением {median_value}")

    # Для категориальных колонок заполним модой
    for col in train_data.select_dtypes(include=['object']).columns:
        if train_data[col].isnull().sum() > 0:
            mode_value = train_data[col].mode()[0]
            train_data[col].fillna(mode_value, inplace=True)
            test_data[col].fillna(mode_value, inplace=True)
            print(f"  Заполнили пропуски в '{col}' модой '{mode_value}'")

# b. Анализ и обработка выбросов
print("\nАнализ выбросов:")
for col in numeric_columns:
    if col == 'cost':  # Не проверяем выбросы в целевой переменной
        continue

    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((train_data[col] < lower_bound) | (train_data[col] > upper_bound)).sum()

    if outliers > 0:
        print(f"  Колонка '{col}': найдено {outliers} выбросов ({outliers / len(train_data) * 100:.2f}%)")

        # Заменяем выбросы граничными значениями
        train_data.loc[train_data[col] < lower_bound, col] = lower_bound
        train_data.loc[train_data[col] > upper_bound, col] = upper_bound

        # То же самое для тестовых данных
        test_data.loc[test_data[col] < lower_bound, col] = lower_bound
        test_data.loc[test_data[col] > upper_bound, col] = upper_bound

        print(f"    Выбросы заменены граничными значениями: [{lower_bound}, {upper_bound}]")

# c. Обработка категориальных переменных
print("\nОбработка категориальных переменных:")
print(f"Всего категориальных переменных: {len(categorical_columns)}")

# Применяем One-Hot кодирование
if len(categorical_columns) > 0:
    print("Применяем One-Hot кодирование")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Обучаем кодировщик на тренировочных данных
    encoded_cats = encoder.fit_transform(train_data[categorical_columns])
    encoded_cats_test = encoder.transform(test_data[categorical_columns])

    # Получаем названия новых столбцов
    feature_names = encoder.get_feature_names_out(categorical_columns)

    # Создаем новые датафреймы с закодированными категориальными признаками
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=feature_names, index=train_data.index)
    encoded_cats_df_test = pd.DataFrame(encoded_cats_test, columns=feature_names, index=test_data.index)

    # Удаляем оригинальные категориальные столбцы и добавляем закодированные
    train_data_encoded = train_data.drop(columns=categorical_columns).join(encoded_cats_df)
    test_data_encoded = test_data.drop(columns=categorical_columns).join(encoded_cats_df_test)

    print(f"После кодирования: {train_data_encoded.shape[1]} признаков")
else:
    train_data_encoded = train_data
    test_data_encoded = test_data

# d. Проверка гипотез
print("\nПроверка гипотез:")

# Гипотеза 1: Чем больше площадь магазина (store_sqft), тем выше стоимость кампании (cost)
print("Гипотеза 1: Стоимость медиа-кампании увеличивается с ростом площади магазина (store_sqft)")
if 'cost' in train_data_encoded.columns and 'store_sqft' in train_data_encoded.columns:
    correlation = train_data_encoded['store_sqft'].corr(train_data_encoded['cost'])
    print(f"Корреляция между store_sqft и cost: {correlation:.2f}")
    if abs(correlation) > 0.3:
        print("Гипотеза подтверждается: есть умеренная или сильная зависимость.")
    else:
        print("Гипотеза не подтверждается: зависимость слабая.")
    # График
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='store_sqft', y='cost', data=train_data_encoded)
    plt.title('store_sqft vs cost')
    plt.tight_layout()
    plt.savefig('hypothesis1_store_sqft_vs_cost.png')
    plt.close()
    print("График сохранён как 'hypothesis1_store_sqft_vs_cost.png'")

# Гипотеза 2: Наличие дополнительных сервисов (например, coffee_bar, video_store) связано с более высокой стоимостью кампании
print("\nГипотеза 2: Наличие дополнительных сервисов увеличивает стоимость кампании")
additional_services = ['coffee_bar', 'video_store', 'salad_bar', 'prepared_food', 'florist']
for service in additional_services:
    if service in train_data_encoded.columns:
        mean_with = train_data_encoded[train_data_encoded[service] == 1]['cost'].mean()
        mean_without = train_data_encoded[train_data_encoded[service] == 0]['cost'].mean()
        diff = mean_with - mean_without
        print(f"Сервис '{service}': средняя cost при наличии = {mean_with:.2f}, без = {mean_without:.2f}, разница = {diff:.2f}")

        plt.figure(figsize=(5, 3))
        sns.boxplot(x=service, y='cost', data=train_data_encoded)
        plt.title(f'{service} vs cost')
        plt.tight_layout()
        plt.savefig(f'hypothesis2_{service}_vs_cost.png')
        plt.close()
        print(f"График сохранён как 'hypothesis2_{service}_vs_cost.png'")

# e. Разделение датасета на трейн и тест
print("\nРазделение данных на тренировочную и тестовую выборки:")
if 'cost' in train_data_encoded.columns:
    X = train_data_encoded.drop(columns=['cost'])
    y = train_data_encoded['cost']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Размер тренировочной выборки: {X_train.shape[0]} строк, {X_train.shape[1]} столбцов")
    print(f"Размер валидационной выборки: {X_val.shape[0]} строк, {X_val.shape[1]} столбцов")
else:
    print("В данных нет столбца 'cost'. Проверьте ваши данные.")
    sys.exit(1)

# 4. Обучение модели MyLineReg
print("\n--- ОБУЧЕНИЕ МОДЕЛИ ---")


# Подготовим функцию для динамического learning rate
def lr_schedule(iteration):
    return 0.1 / (1 + 0.01 * iteration)


# Создаем экземпляр модели
model = MyLineReg(
    n_iter=100,
    learning_rate=lr_schedule,
    metric='mse',
    reg='l2',
    l2_coef=0.01,
    sgd_sample=0.5,  # Стохастический градиентный спуск на 50% выборки
    random_state=42,
    normalize=True
)

# Обучаем модель
print("Обучаем модель MyLineReg...")
model.fit(X_train, y_train, verbose=10)  # Выводим прогресс каждые 10 итераций

# Загружаем лучшие веса, найденные во время обучения
model.load_best_weights()

# Делаем предсказания на валидационной выборке
print("\nПредсказания на валидационной выборке:")
y_pred_val = model.predict(X_val)

# Оцениваем качество модели
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred_val)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Визуализируем прогресс обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(model.history_errors)
plt.title('MSE во время обучения')
plt.xlabel('Итерация')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(model.history_metric)
plt.title(f'{model.metric} во время обучения')
plt.xlabel('Итерация')
plt.ylabel(model.metric)

plt.tight_layout()
plt.savefig('training_progress.png')
plt.close()
print("Графики процесса обучения сохранены в файле 'training_progress.png'")

# Важность признаков (по абсолютной величине весов)
coefs = model.get_coef()
feature_importance = pd.DataFrame({'Признак': X.columns, 'Важность': np.abs(coefs)})
feature_importance = feature_importance.sort_values('Важность', ascending=False)

print("\nТоп-10 важных признаков:")
print(feature_importance.head(10))

# Визуализация важности признаков
plt.figure(figsize=(12, 8))
sns.barplot(x='Важность', y='Признак', data=feature_importance.head(20))
plt.title('Топ-20 важных признаков')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("График важности признаков сохранен в файле 'feature_importance.png'")

print("\n--- ГОТОВО ---")