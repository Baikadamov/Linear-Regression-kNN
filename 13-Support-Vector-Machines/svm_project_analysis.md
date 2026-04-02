# Разбор кода: SVM Project — Wine Fraud Detection

**Задача**: По химическим характеристикам вина (pH, кислотность, алкоголь и др.) определить, является ли вино легальным (Legit) или поддельным (Fraud).

---

## 1. Импорт библиотек

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

- `numpy` — работа с массивами и математическими операциями
- `pandas` — загрузка и обработка табличных данных (DataFrame)
- `seaborn` — красивые статистические графики поверх matplotlib
- `matplotlib.pyplot` — базовая библиотека для построения графиков
- `%matplotlib inline` — магическая команда Jupyter, чтобы графики отображались прямо в ноутбуке

---

## 2. Загрузка и осмотр данных

```python
df = pd.read_csv("wine_fraud.csv")
df.head()
```

- `pd.read_csv("wine_fraud.csv")` — читает CSV-файл и создаёт DataFrame `df`
- `.head()` — показывает первые 5 строк таблицы для быстрого осмотра структуры данных

```python
df.info()
```

- Выводит информацию о каждом столбце: количество непустых значений и тип данных
- Результат: 11 числовых столбцов (`float64`) и 2 текстовых (`str`), всего 6497 строк, пропусков нет

```python
df.describe()
```

- Статистика по числовым столбцам: среднее (`mean`), стандартное отклонение (`std`), минимум, максимум, квартили (25%, 50%, 75%)
- Помогает увидеть разброс данных и возможные выбросы (например, `residual sugar` max=65.8 при среднем 5.4)

---

## 3. EDA — Исследовательский анализ данных

### Распределение классов

```python
df["quality"].value_counts()
```

- Считает количество каждого уникального значения в столбце `quality`
- Результат: Legit = 6251, Fraud = 246 — **сильный дисбаланс классов** (мошеннических вин всего ~3.8%)

```python
df["quality"].value_counts(normalize=True)
```

- `normalize=True` — показывает доли вместо абсолютных чисел
- Legit = 96.2%, Fraud = 3.8%

### Countplot: количество Legit и Fraud

```python
plt.figure(figsize=(6,4))
sns.countplot(x="quality", data=df, palette="Set2")
plt.title("Распределение классов (Legit vs Fraud)")
plt.show()
```

- `plt.figure(figsize=(6,4))` — создаёт фигуру размером 6×4 дюймов
- `sns.countplot(x="quality", data=df)` — столбчатая диаграмма, автоматически считает количество каждого класса
- `palette="Set2"` — цветовая палитра для столбцов
- `plt.title(...)` — заголовок графика
- `plt.show()` — отрисовать и показать график

### Countplot: Legit/Fraud по типу вина

```python
plt.figure(figsize=(8,5))
sns.countplot(x="type", data=df, hue="quality", palette="Set2")
plt.title("Распределение Fraud/Legit по типу вина")
plt.show()
```

- `x="type"` — ось X показывает тип вина (red/white)
- `hue="quality"` — разделяет столбцы по классу (Legit/Fraud) разными цветами
- Показывает, как распределены подделки между красным и белым вином

### Boxplot ключевых признаков

```python
features = ["fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "alcohol"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(x="quality", y=feat, data=df, ax=ax, palette="Set2")
    ax.set_title(feat)
plt.tight_layout()
plt.show()
```

- `features` — список из 6 ключевых признаков для визуализации
- `plt.subplots(2, 3, figsize=(15, 8))` — создаёт сетку 2 строки × 3 столбца = 6 графиков
- `axes.flatten()` — превращает 2D-массив осей в плоский одномерный список для удобного перебора в цикле
- `zip(axes.flatten(), features)` — связывает каждый график (ax) с одним признаком (feat)
- `sns.boxplot(x="quality", y=feat, ...)` — ящик с усами: показывает медиану, квартили и выбросы для Legit vs Fraud
- `ax.set_title(feat)` — подписывает каждый подграфик названием признака
- `plt.tight_layout()` — автоматически подгоняет расстояния между графиками, чтобы не перекрывались

### Средние значения по классам

```python
df.groupby("quality").mean(numeric_only=True)
```

- `groupby("quality")` — группирует данные по классу (Fraud / Legit)
- `.mean(numeric_only=True)` — считает средние значения только для числовых столбцов
- Видно, что у Fraud выше volatile acidity (0.47 vs 0.33) и chlorides (0.062 vs 0.056), ниже free sulfur dioxide (22.9 vs 30.8)

### Корреляция с целевой переменной

```python
df_corr = df.copy()
df_corr["quality_num"] = (df_corr["quality"] == "Fraud").astype(int)
corr = df_corr.drop(columns=["quality","type"]).corr()[["quality_num"]].sort_values("quality_num", ascending=False)
print(corr)
```

- `df.copy()` — создаёт копию DataFrame, чтобы не менять оригинал
- `(df_corr["quality"] == "Fraud").astype(int)` — превращает строку "Fraud" в 1, "Legit" в 0 (бинарная переменная)
- `.corr()` — вычисляет матрицу корреляций Пирсона между всеми числовыми столбцами
- `[["quality_num"]]` — берём только столбец корреляций с целевой переменной
- `.sort_values("quality_num", ascending=False)` — сортируем по убыванию
- Результат: volatile acidity (+0.15) — наиболее коррелирует с мошенничеством

### Clustermap корреляций

```python
plt.figure()
sns.clustermap(df.drop(columns=["quality","type"]).corr(), annot=True, fmt=".2f", figsize=(10,8))
plt.show()
```

- `sns.clustermap(...)` — тепловая карта корреляций с **иерархической кластеризацией** (автоматически перегруппировывает похожие признаки рядом)
- `annot=True` — показывает числовые значения на каждой ячейке
- `fmt=".2f"` — формат числа: 2 знака после запятой
- `figsize=(10,8)` — размер карты

---

## 4. Предобработка данных

### One-Hot кодирование столбца type

```python
df = pd.get_dummies(df, columns=["type"], drop_first=True)
df.head()
```

- `pd.get_dummies(df, columns=["type"])` — One-Hot кодирование: текстовый столбец `type` (red/white) превращается в числовой
- `drop_first=True` — убирает один dummy-столбец, чтобы избежать мультиколлинеарности. Остаётся только `type_white`: если `type_white=0` → red, `type_white=1` → white

### Формирование X и y

```python
X = df.drop(columns=["quality"])
y = df["quality"]
```

- `X` — матрица признаков (features): все столбцы **кроме** целевого `quality`
- `y` — целевая переменная (target): столбец `quality` (Legit / Fraud)

### Разбивка на train/test

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)
```

- `train_test_split(X, y, ...)` — разбивает данные на обучающую и тестовую выборки
- `test_size=0.10` — 10% данных уходит в тест, 90% в обучение
- `random_state=101` — фиксирует генератор случайных чисел для воспроизводимости результатов
- Возвращает 4 переменные: `X_train`, `X_test` (признаки), `y_train`, `y_test` (метки)

### Масштабирование данных

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

- `StandardScaler()` — стандартизация: каждый признак приводится к среднему = 0 и стандартному отклонению = 1
- `fit_transform(X_train)` — вычисляет среднее и стд. отклонение **по обучающей выборке** и сразу трансформирует её
- `transform(X_test)` — применяет **те же самые** параметры (среднее и стд) к тестовой выборке
- **Важно**: нельзя вызывать `fit` на тестовых данных, иначе произойдёт утечка данных (data leakage)
- SVM чувствителен к масштабу признаков, поэтому масштабирование обязательно

---

## 5. Обучение базовой модели SVM

```python
from sklearn.svm import SVC
svc_base = SVC()
svc_base.fit(X_train, y_train)
```

- `SVC()` — Support Vector Classifier (метод опорных векторов) с параметрами по умолчанию:
  - `kernel='rbf'` — радиально-базисная функция (нелинейное ядро)
  - `C=1.0` — параметр регуляризации
  - `gamma='scale'` — масштаб ядра, автоматически = 1 / (n_features × var(X))
- `.fit(X_train, y_train)` — обучает модель: ищет оптимальную разделяющую гиперплоскость

### Оценка базовой модели

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
y_pred_base = svc_base.predict(X_test)
print(confusion_matrix(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))
```

- `predict(X_test)` — предсказания модели для тестовых данных
- `confusion_matrix(y_test, y_pred_base)` — матрица ошибок:
  - Результат: `[[0, 27], [0, 623]]`
  - 0 Fraud найдено правильно, 27 Fraud пропущено, 0 ложных Fraud, 623 Legit верно
- `classification_report(...)` — precision, recall, F1-score для каждого класса
  - Fraud: precision=0.00, recall=0.00, F1=0.00 — модель **вообще не находит** мошенничество
  - Legit: precision=0.96, recall=1.00, F1=0.98
  - Accuracy=96% — обманчиво высокая, модель просто всё называет Legit

---

## 6. GridSearchCV — подбор гиперпараметров

### Определение сетки параметров

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"]
}
```

- `param_grid` — словарь с параметрами для перебора:
  - **C** (регуляризация): чем больше, тем жёстче граница, меньше допуск ошибок на обучении
  - **gamma** (коэффициент ядра RBF): чем больше, тем сложнее/изогнутее граница решения
  - **kernel**: фиксирован `rbf`
  - Всего 4 × 4 × 1 = **16 комбинаций**

### Запуск GridSearchCV

```python
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5, scoring="f1_macro")
grid.fit(X_train, y_train)
```

- `GridSearchCV(...)` — полный перебор всех комбинаций параметров с кросс-валидацией
- `SVC()` — базовая модель, на основе которой строятся все варианты
- `cv=5` — 5-fold кросс-валидация: данные делятся на 5 частей, модель обучается на 4 и проверяется на 1, процесс повторяется 5 раз
- `scoring="f1_macro"` — метрика оптимизации: **средний F1 по обоим классам** (не accuracy, т.к. данные несбалансированы)
- `refit=True` — после поиска лучших параметров автоматически переобучает модель на **всех** тренировочных данных
- `verbose=2` — подробный вывод каждого шага обучения
- Итого: 16 комбинаций × 5 фолдов = **80 обучений модели**

### Лучшие параметры

```python
print("Лучшие параметры:", grid.best_params_)
print("Лучший score (cv):", round(grid.best_score_, 4))
```

- `grid.best_params_` — лучшая комбинация: `C=100, gamma=0.1, kernel=rbf`
- `grid.best_score_` — лучший F1 macro на кросс-валидации: **0.6162**

---

## 7. Оценка лучшей модели

### Предсказания и матрица ошибок

```python
y_pred_best = grid.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
```

- `grid.predict(X_test)` — предсказания лучшей модели (автоматически использует `refit`-модель)
- Confusion matrix: `[[2, 25], [5, 618]]`:
  - **2** Fraud правильно найдены (True Positive)
  - **25** Fraud пропущены (False Negative)
  - **5** Legit ошибочно названы Fraud (False Positive)
  - **618** Legit верно определены (True Negative)

### Визуализация confusion matrix

```python
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit","Fraud"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — SVM (лучшая модель)")
plt.show()
```

- `ConfusionMatrixDisplay(...)` — объект для визуализации матрицы ошибок
- `display_labels=["Legit","Fraud"]` — подписи классов
- `.plot(cmap="Blues")` — отрисовка с синей цветовой картой
- Показывает наглядно, сколько объектов каждого класса были классифицированы правильно/неправильно

### Classification report

```python
print(classification_report(y_test, y_pred_best))
```

- **Fraud**: precision=0.29, recall=0.07, F1=0.12 — модель плохо ловит мошенничество (нашла только 2 из 27)
- **Legit**: precision=0.96, recall=0.99, F1=0.98 — легальные вина определяются отлично
- **Accuracy** = 0.95 — высокая, но не показательна из-за дисбаланса классов

---

## 8. Общие выводы

1. **Дисбаланс классов**: Fraud встречается лишь в ~3.8% случаев — задача сильно несбалансированная, поэтому accuracy не показательна, важны precision и recall для класса Fraud
2. **Базовая SVC** даёт accuracy 96%, но recall для Fraud = 0% — модель просто предсказывает всё как Legit
3. **GridSearchCV** подобрал оптимальные `C=100` и `gamma=0.1`, немного улучшив recall для Fraud (с 0% до 7%)
4. **Важные признаки**: volatile acidity (+0.15 корреляция) и chlorides (+0.03) — наиболее связаны с мошенничеством
5. **Возможные улучшения**:
   - Использовать `class_weight='balanced'` в SVC для учёта дисбаланса
   - Применить oversampling (SMOTE) для увеличения количества Fraud-примеров
   - Попробовать другие модели (Random Forest, XGBoost)
   - Оптимизировать по recall вместо F1 macro
