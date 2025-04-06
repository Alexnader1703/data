import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import matplotlib.colors as mcolors

# Для обработки русского текста
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)

# Инициализация компонентов Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)


# Функция для загрузки песен из файла
def load_songs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Разделение песен по маркеру
    songs = content.split("---НОВАЯ ПЕСНЯ---")

    # Удаление пустых строк
    songs = [song.strip() for song in songs if song.strip()]

    return songs


# Функция для лемматизации текста с помощью natasha
def lemmatize_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    return [token.lemma for token in doc.tokens]


# Функция для препроцессинга текста
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление знаков препинания и цифр
    text = re.sub(r'[^\w\s]|[\d]', ' ', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Функция для удаления стоп-слов
def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if token not in stopwords]


# Функция для визуализации наиболее частых слов
def plot_wordcloud(text_data):
    all_text = ' '.join([' '.join(text) for text in text_data])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        contour_width=3,
        contour_color='steelblue',
        font_path='arial.ttf'  # Укажите путь к шрифту с поддержкой кириллицы
    ).generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud наиболее частых слов')
    plt.show()


# Функция для поиска похожих слов в модели Word2Vec
def show_similar(model, word, topn=10):
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        print(f"Слова, похожие на '{word}':")
        for w, score in similar_words:
            print(f"{w}: {score:.4f}")
    except KeyError:
        print(f"Слово '{word}' не найдено в словаре модели.")


# Функция для визуализации слов с помощью t-SNE
def visualize_tsne(model, frequent_words):
    # Получение векторов слов
    word_vectors = np.array([model.wv[word] for word in frequent_words if word in model.wv])
    words = [word for word in frequent_words if word in model.wv]

    # Применение t-SNE для снижения размерности
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    vectors_2d = tsne.fit_transform(word_vectors)

    # Визуализация
    plt.figure(figsize=(12, 8))

    # Создание цветовой палитры
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, word in enumerate(words):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], c=colors[i % len(colors)])
        plt.annotate(word, vectors_2d[i], fontsize=12)

    plt.title('t-SNE визуализация векторов слов')
    plt.grid(True)
    plt.show()


# Основная функция анализа
def analyze_songs(file_path):
    # Загрузка песен
    print("Загрузка песен...")
    songs = load_songs(file_path)
    print(f"Загружено {len(songs)} песен")

    # Русские стоп-слова
    russian_stopwords = [
        'и', 'в', 'на', 'с', 'по', 'для', 'не', 'что', 'это', 'так',
        'к', 'у', 'я', 'ты', 'он', 'она', 'они', 'мы', 'вы', 'все',
        'как', 'но', 'да', 'от', 'до', 'из', 'о', 'же', 'за', 'бы',
        'та', 'то', 'эта', 'эти', 'этот', 'а', 'б', 'в', 'г', 'д',
        'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р',
        'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь',
        'э', 'ю', 'я', 'есть', 'был', 'была', 'были', 'быть', 'будет',
        'будут', 'тебя', 'меня', 'себя', 'его', 'её', 'их', 'мой',
        'твой', 'свой', 'ваш', 'наш', 'мне', 'тебе', 'ему', 'нам',
        'вам', 'им', 'ним', 'нем', 'вот', 'этом', 'том','лай','лала','весь','тот','ла','лай','лаа','ла','кто','нет','ни','над','под','чей'
    ]

    # Препроцессинг песен
    print("Препроцессинг текста...")
    processed_songs = []
    tokenized_songs = []

    for song in songs:
        # Препроцессинг текста
        processed_text = preprocess_text(song)

        # Лемматизация
        lemmatized_tokens = lemmatize_text(processed_text)

        # Удаление стоп-слов
        clean_tokens = remove_stopwords(lemmatized_tokens, russian_stopwords)

        processed_songs.append(' '.join(clean_tokens))
        tokenized_songs.append(clean_tokens)

    # Сохранение обработанных песен в файл
    with open('processed_songs.txt', 'w', encoding='utf-8') as f:
        for i, song in enumerate(processed_songs):
            f.write(f"Песня {i + 1}:\n{song}\n\n")

    print("Обработанные песни сохранены в файл 'processed_songs.txt'")

    # TF-IDF анализ
    print("\nАнализ TF-IDF...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_songs)

    # Получение слов и их TF-IDF значений
    feature_names = vectorizer.get_feature_names_out()

    # Анализ наиболее частых слов
    all_tokens = [token for song_tokens in tokenized_songs for token in song_tokens]
    word_freq = Counter(all_tokens)
    top_words = word_freq.most_common(15)

    print("\nНаиболее частые слова:")
    for word, count in top_words:
        print(f"{word}: {count}")

    # Построение WordCloud
    print("\nПостроение WordCloud...")
    plot_wordcloud(tokenized_songs)

    # Подготовка данных для Word2Vec
    print("\nОбучение модели Word2Vec...")
    w2v_model = Word2Vec(tokenized_songs, vector_size=100, window=5, min_count=2, workers=4, epochs=20)

    # Проверка близких слов
    if top_words:
        example_word = top_words[0][0]
        print(f"\nПроверка слов, близких к '{example_word}':")
        show_similar(w2v_model, example_word)

    # Визуализация t-SNE
    print("\nВизуализация t-SNE для наиболее частых слов...")
    frequent_words = [word for word, _ in top_words]
    visualize_tsne(w2v_model, frequent_words)

    return tokenized_songs, w2v_model


# Запуск анализа
if __name__ == "__main__":
    file_path = "songs.txt"  # Путь к файлу с песнями
    tokenized_songs, w2v_model = analyze_songs(file_path)

    print("\nАнализ завершен!")