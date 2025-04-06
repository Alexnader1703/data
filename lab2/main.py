import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from lin_regres import MyLineReg
df= sns.load_dataset("mpg")
dfd=df.dropna()
import matplotlib.pyplot as plt
def stat():
    print(f"Строк:{df.shape[0]}\nСтолбцов:{df.shape[1]}")

    num_df=df.select_dtypes(include=['number'])
    print(df.columns)
    eda_res= pd.DataFrame({
        "Пропущенные значения (%)": num_df.isnull().mean() * 100,
        "Минимум": num_df.min(),
        "Максимум": num_df.max(),
        "Среднее": num_df.mean(),
        "Медиана": num_df.median(),
        "Дисперсия": num_df.var(),
        "Квантиль 0.1": num_df.quantile(0.1),
        "Квантиль 0.9": num_df.quantile(0.9),
        "Квартиль 1 (25%)": num_df.quantile(0.25),
        "Квартиль 3 (75%)": num_df.quantile(0.75)
    })
    print(df.dtypes)
    ctg_df = df.select_dtypes(include=['object'])
    catg_res = pd.DataFrame({
        "Пропущенные значения (%)": ctg_df.isnull().mean() * 100,
        "Уникальных значений": ctg_df.nunique(),
        "Мода (наиболее частое значение)": ctg_df.mode().iloc[0]
    })
    print(eda_res)
    print(catg_res)

def gipUSA():
    usa_mpg = dfd[dfd["origin"] == "usa"]["mpg"]
    europe_mpg = dfd[dfd["origin"] == "europe"]["mpg"]

    t_stat, p_value = stats.ttest_ind(usa_mpg, europe_mpg, alternative="less")

    print("Гипотеза 1: Машины из USA менее экономичны")
    print(f"T-статистика: {t_stat:.3f}, p-value: {p_value:.3f}")

    if p_value < 0.05:
        print("Машины из США менее экономичны.")
    else:
        print("Статистически значимых различий нет.")


def gipCil():
    
    groups = [dfd[dfd["cylinders"] == c]["horsepower"] for c in dfd["cylinders"].unique()]

    f_stat, p_value = stats.f_oneway(*groups)

    print("\nГипотеза 2: Число цилиндров влияет на мощность")
    print(f"F-статистика: {f_stat:.3f}, p-value: {p_value:.3f}")

    if p_value < 0.05:
        print("Мощность зависит от числа цилиндров.")
    else:
        print("Связи между цилиндрами и мощностью нет.")


def encod():
    dfe = pd.get_dummies(df, columns=["origin"], drop_first=True)
    print(dfe)

def kor():
    target = "mpg"
    features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]

    correlation_matrix = dfd[features + [target]].corr()

    print(correlation_matrix)
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Корреляционная матрица признаков и целевого столбца")
    plt.show()
def grad():
    X = dfd[["horsepower"]]
    y = dfd["mpg"]

    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    model = MyLineReg(learning_rate=0.0001, n_iter=10000, normalize=False)
    model.fit(X, y, verbose=100)

    plt.plot(model.history_errors)
    plt.xlabel("Итерации")
    plt.ylabel("MSE")
    plt.title("График потерь")
    plt.show()


    y_pred = model.predict(X)
    print(y_pred)

stat()
gipUSA()
gipCil()
encod()
kor()
grad()

