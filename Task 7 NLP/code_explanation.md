# Разбор кода: `sentiment_analysis_assignment.ipynb`

Ноутбук идёт сверху вниз: загрузка → EDA → препроцессинг → split → векторизация → 4 модели → лидерборд → анализ лучшей.

---

## Импорты

```python
import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import nltk
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
```

- `re`, `string` — регулярки и список знаков пунктуации для очистки текста.
- `CountVectorizer` — BoW (Bag of Words).
- `TfidfVectorizer` — TF-IDF, та же логика что и BoW, но со взвешиванием.
- `LabelEncoder` — `negative/neutral/positive` → `0/1/2` (XGBoost требует числовые метки).
- `XGBClassifier`, `CatBoostClassifier` — наши две модели.
- `stopwords` — список английских стоп-слов из NLTK.
- `RANDOM_STATE = 42` — глобальный seed для воспроизводимости.

---

## 1. Загрузка и EDA

```python
df = pd.read_csv("sentiment_analysis.csv")
print("Shape:", df.shape)
df.head()
```

Простая загрузка CSV. После запуска получаем форму `(499, 7)`.

```python
df.info()
```

Сводка по типам столбцов. Видно что `text` и `sentiment` — единственные что нам нужны.

```python
print("Missing values:")
print(df.isnull().sum())
df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)
df["sentiment"] = df["sentiment"].str.strip().str.lower()
print("\nClass distribution:")
print(df["sentiment"].value_counts())
```

**Построчно:**
- `df.isnull().sum()` — сколько NaN в каждой колонке.
- `dropna(subset=["text", "sentiment"])` — удаляем строки, где НЕТ текста или метки. Остальные NaN (в `Time of Tweet` и т.п.) нам безразличны.
- `.reset_index(drop=True)` — пересобираем индексы после удаления.
- `.str.strip().str.lower()` — нормализуем метки. В оригинальном CSV есть пробелы (`" Twitter "`) — для текста они нестрашны, но для меток лучше почистить.
- `value_counts()` — распределение классов.

```python
plt.figure(figsize=(6, 6))
counts = df["sentiment"].value_counts()
colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
plt.pie(
    counts, labels=counts.index, autopct="%1.1f%%",
    colors=[colors.get(c, "gray") for c in counts.index],
    startangle=90,
)
plt.title("Sentiment Distribution (3 classes)")
plt.show()
```

Pie-chart с осмысленными цветами: позитив зелёный, нейтрал серый, негатив красный. Хороший способ показать на одном графике распределение классов.

---

## 2. Препроцессинг

```python
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
```

- `nltk.download("stopwords")` — скачивает список стоп-слов при первом запуске.
- `quiet=True` — гасит вывод про загрузку.
- `set(...)` — превращаем список в set для быстрого `in`-поиска (O(1) вместо O(N)).

```python
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)
```

**Построчно (что делает каждый regex):**

| Строка | Что чистит | Пример |
|--------|-----------|--------|
| `text.lower()` | регистр | `Great!` → `great!` |
| `r"http\S+|www\S+|https\S+"` | ссылки | `Check https://x.co` → `Check ` |
| `r"@\w+"` | упоминания | `Hi @user!` → `Hi !` |
| `r"#"` | решётки (слово оставляем) | `#happy` → `happy` |
| `r"\d+"` | числа | `On day 5` → `On day ` |
| `str.maketrans("", "", string.punctuation)` | вся пунктуация | `wow!` → `wow` |
| `r"\s+"` → `" "` + `.strip()` | лишние пробелы | `  hi   you  ` → `hi you` |
| фильтр токенов | стоп-слова + слова ≤ 2 букв | `the is a great` → `great` |

`str.maketrans` создаёт таблицу замены: первые два аргумента — это пары "из чего в что" (пусто), третий — символы для удаления.

```python
df["clean_text"] = df["text"].apply(preprocess_text)
df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
```

- `.apply(preprocess_text)` — применяет функцию к каждой строке (медленно, но удобно).
- `df["clean_text"].str.len() > 0` — отбрасываем строки, где после очистки ничего не осталось (например, если изначально текст был "??!!" — после очистки пусто).

---

## 3. Train/Test split

```python
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["sentiment"])
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
```

- `LabelEncoder.fit_transform` — присваивает каждому уникальному значению целое число (по алфавиту):
  - `negative` → 0
  - `neutral` → 1
  - `positive` → 2
- XGBoost требует именно числовые метки. Строки не примет.

```python
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.25,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)
```

- `test_size=0.25` — 25% на тест.
- `stratify=df["label"]` — **критично!** Сохраняет пропорции классов в train и test. Без этого случайный сплит может дать перекос.
- `random_state=42` — фиксирует random seed.

**Важно:** делим **текст до векторизации**, не после. Так мы избегаем утечки данных: словарь векторизатора строится только по train, тестовые слова, которых нет в трейне, просто игнорируются.

---

## 4. Векторизация

```python
count_vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
X_train_bow = count_vec.fit_transform(X_train_text)
X_test_bow = count_vec.transform(X_test_text)
```

- `ngram_range=(1, 2)` — берём и одиночные слова (unigrams), и пары (bigrams). `not good` станет отдельным признаком, отличающимся от `good`.
- `min_df=2` — слово/биграмма должна встретиться минимум в 2 документах трейна. Отсеивает опечатки и редкий шум.
- `fit_transform` на трейне — строит словарь + векторизует. `transform` на тесте — использует тот же словарь, новые слова игнорируются.

```python
tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf_vec.fit_transform(X_train_text)
X_test_tfidf = tfidf_vec.transform(X_test_text)
```

То же самое, но с TF-IDF взвешиванием.

Обратить внимание: оба векторизатора возвращают **sparse матрицы** (`scipy.sparse.csr_matrix`), а не numpy-массивы. Это потому что большинство значений — нули, и хранить их явно расточительно. XGBoost и CatBoost умеют работать со sparse напрямую.

---

## 5. Обучение 4 моделей

```python
def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "weighted_f1": f1,
    }
```

Хелпер: обучает, предсказывает, считает метрики. Возвращает dict со всем нужным (модель, предсказания, метрики) — чтобы потом не переобучать заново.

```python
results = []

results.append(evaluate(
    "BoW + XGBoost",
    XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                  random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1),
    X_train_bow, X_test_bow, y_train, y_test,
))
```

**XGBClassifier параметры:**
- `n_estimators=300` — 300 деревьев в ансамбле.
- `max_depth=6` — глубина каждого дерева. Больше → сложнее, переобучается.
- `learning_rate=0.1` — насколько сильно каждое новое дерево корректирует предыдущие.
- `eval_metric="mlogloss"` — для мультикласса (вместо binary logloss).
- `n_jobs=-1` — использовать все ядра CPU.

```python
results.append(evaluate(
    "BoW + CatBoost",
    CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                       random_state=RANDOM_STATE, verbose=False),
    X_train_bow, X_test_bow, y_train, y_test,
))
```

**CatBoostClassifier параметры:**
- `iterations=300` — 300 итераций (= 300 деревьев).
- `depth=6` — глубина.
- `verbose=False` — иначе CatBoost выводит прогресс каждые 50 итераций.

Аналогично делаем TF-IDF + XGBoost и TF-IDF + CatBoost.

В конце выводим результаты:
```python
for r in results:
    print(f"{r['name']:20s} | Accuracy = {r['accuracy']:.4f} | Weighted F1 = {r['weighted_f1']:.4f}")
```

- `f"{r['name']:20s}"` — выравнивание по 20 символам.
- `{:.4f}` — 4 знака после запятой.

---

## 6. Лидерборд

```python
leaderboard = pd.DataFrame([
    {"Combination": r["name"], "Accuracy": r["accuracy"], "Weighted F1": r["weighted_f1"]}
    for r in results
]).sort_values("Weighted F1", ascending=False).reset_index(drop=True)
leaderboard.index = leaderboard.index + 1
leaderboard
```

- List comprehension собирает данные в список dicts → `pd.DataFrame` → таблица.
- `sort_values("Weighted F1", ascending=False)` — сортируем по F1 от большего к меньшему.
- `leaderboard.index = leaderboard.index + 1` — нумерация мест с 1, не с 0.

### Bar chart

```python
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(leaderboard))
width = 0.35
ax.bar(x - width / 2, leaderboard["Accuracy"], width, label="Accuracy", color="#3498db")
ax.bar(x + width / 2, leaderboard["Weighted F1"], width, label="Weighted F1", color="#e67e22")
```

- Группированные бары: для каждой комбинации две колонки (Accuracy и F1) рядом.
- `x - width/2` и `x + width/2` — смещение левой и правой колонки от центральной позиции.

```python
for i, (acc, f1) in enumerate(zip(leaderboard["Accuracy"], leaderboard["Weighted F1"])):
    ax.text(i - width / 2, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=9)
    ax.text(i + width / 2, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=9)
```

Подписи числами над каждым баром. `ha="center"` — горизонтальное выравнивание по центру.

---

## 7. Анализ лучшей модели

```python
best = max(results, key=lambda r: r["weighted_f1"])
print(f"Best combination: {best['name']}")
```

- `max(..., key=lambda r: r["weighted_f1"])` — берём элемент с максимальным F1. Поскольку каждый `r` это dict, нужен `key`-функция, чтобы сказать "сравнивать по полю `weighted_f1`".

```python
print(classification_report(
    y_test, best["y_pred"], target_names=label_encoder.classes_,
))
```

- `classification_report` — таблица precision/recall/f1/support по каждому классу.
- `target_names=label_encoder.classes_` — показывать имена `negative/neutral/positive` вместо `0/1/2`.

```python
cm = confusion_matrix(y_test, best["y_pred"])
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.xlabel("Predicted"); plt.ylabel("Actual")
```

- `confusion_matrix` — матрица `actual × predicted`.
- `annot=True, fmt="d"` — печатать числа в клетках, целые.
- `cmap="Blues"` — синяя цветовая шкала.
- Диагональ = правильные предсказания. Off-diagonal = ошибки.

---

## 8. Custom predictions

```python
best_vectorizer = tfidf_vec if "TF-IDF" in best["name"] else count_vec
best_model = best["model"]

samples = [
    "I absolutely love this product, it changed my life!",
    "This is the worst experience ever. Total waste of money.",
    "The package arrived on Tuesday.",
    "Не самая лучшая идея на свете, но и не худшая.",
]

samples_clean = [preprocess_text(s) for s in samples]
samples_vec = best_vectorizer.transform(samples_clean)
samples_pred = np.asarray(best_model.predict(samples_vec)).ravel().astype(int)
samples_labels = label_encoder.inverse_transform(samples_pred)

for raw, clean, label in zip(samples, samples_clean, samples_labels):
    print(f"Text : {raw}")
    print(f"Clean: {clean}")
    print(f"Pred : {label}")
    print()
```

**Тонкости:**
- `best_vectorizer = tfidf_vec if "TF-IDF" in best["name"] else count_vec` — выбираем тот векторизатор, на котором тренировалась лучшая модель.
- `np.asarray(...).ravel().astype(int)` — CatBoost возвращает predictions как 2D массив `[[0], [1], ...]`. `ravel()` сплющивает в 1D, `astype(int)` гарантирует целые числа для `inverse_transform`.
- `label_encoder.inverse_transform(samples_pred)` — обратное превращение `0/1/2` → `negative/neutral/positive`.

---

## Локальный запуск

```bash
source .venv/bin/activate
cd "Task 7 NLP"
jupyter nbconvert --to notebook --execute --inplace sentiment_analysis_assignment.ipynb
```

Прогон ~30 секунд. Все 4 модели обучаются последовательно — самая медленная часть это CatBoost (~10 секунд × 2 раза).

## Фактические числа прогона

- Train/Test: 372 / 124 примера.
- Label mapping: `{negative: 0, neutral: 1, positive: 2}`.
- **Best: TF-IDF + CatBoost** — Accuracy 0.5968, Weighted F1 0.5789.
- Per-class F1: negative 0.35 / neutral 0.66 / positive 0.67.

Полный разбор результатов и интерпретация — в `explanation.md`, секция "Лидерборд".
