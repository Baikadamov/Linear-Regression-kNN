# Разбор кода: PCA + HDBSCAN + KMeans

Ноутбук `narxoz-aml-clustering-pca-anomalydet.ipynb` идёт по шагам: загрузка → preprocessing → PCA → HDBSCAN → KMeans → сравнение.

---

## Импорты

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import hdbscan
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
```

- `pandas` — таблицы (DataFrame), удобный API для CSV
- `numpy` — векторы и матрицы
- `matplotlib.pyplot` + `seaborn` — построение графиков
- `StandardScaler` — стандартизация (вычитание среднего, деление на σ)
- `LabelEncoder` — кодирование строковых категорий в числа
- `PCA` — метод главных компонент
- `hdbscan` — внешняя библиотека (не входит в sklearn)
- `KMeans` — алгоритм центроид-кластеризации
- `warnings.filterwarnings("ignore")` — гасит предупреждения чтобы вывод был чище

---

## Загрузка датасета

```python
df = pd.read_csv("Train_data.csv")
df.info()
```

- `pd.read_csv(...)` — читает CSV-файл и возвращает DataFrame
- Путь относительный — файл лежит рядом с ноутбуком. Если запускаешь на Kaggle, нужно вернуть исходный `/kaggle/input/datasets/sampadab17/network-intrusion-detection/Train_data.csv`.
- `df.info()` — сводка: количество строк, типы столбцов, наличие пропусков

**Результат:** 25192 строки × 42 столбца. 23 int64, 15 float64, 4 object (строковые).

---

## Просмотр данных

```python
df.head()
```

Показывает первые 5 строк. Полезно глазами проверить что данные загрузились правильно.

```python
assert df.isnull().sum().sum().astype(int) == 0, "There are missing data!"
```

- `df.isnull()` — матрица True/False где пропуски
- `.sum().sum()` — суммирует по столбцам, потом сворачивает в одно число
- `assert ... == 0` — если пропусков нет, идём дальше. Иначе — ошибка с сообщением

---

## Визуализация распределения классов

```python
df["class"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(6, 6),
    ylabel=""
)
plt.title("Class Distribution")
plt.show()
```

- `df["class"]` — обращение к столбцу (Series)
- `.value_counts()` — считает сколько раз встречается каждое значение → Series `{normal: N1, anomaly: N2}`
- `.plot(kind="pie")` — круговая диаграмма
- `autopct="%1.1f%%"` — подписи в процентах с 1 знаком после запятой
- `ylabel=""` — убирает подпись оси Y (по умолчанию pandas пишет имя столбца)

---

## Разделение признаков и целевой переменной

```python
target = df["class"]
X = df.drop(columns=["class"])
```

- `target` — столбец `class` отдельно (он понадобится для сравнения с реальной разметкой)
- `X` — все признаки кроме `class`. `df.drop(columns=[...])` создаёт **новый** DataFrame без указанных столбцов (не меняет оригинал)

---

## Поиск категориальных столбцов

```python
categorical_cols = X.select_dtypes(include=["object"]).columns
print("Categorical columns:")
print(categorical_cols)
```

- `select_dtypes(include=["object"])` — выбирает только столбцы со строками (тип object в pandas)
- `.columns` — возвращает индекс с именами этих столбцов

**Результат:** `['protocol_type', 'service', 'flag']` — три категориальных признака.

---

## Кодирование категорий

```python
for col in categorical_cols:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
```

**Построчно:**
- `for col in categorical_cols` — перебираем имена категориальных столбцов
- `encoder = LabelEncoder()` — создаём новый кодировщик для каждого столбца
- `encoder.fit_transform(X[col])` — подгоняет кодировщик и применяет:
  - `fit` — находит уникальные значения и присваивает им номера: `tcp` → 0, `udp` → 1, `icmp` → 2
  - `transform` — заменяет строки на эти номера
- `X[col] = ...` — заменяет столбец в DataFrame

**Важно:** создаём отдельный `LabelEncoder` для каждого столбца. Если использовать один на все — он перепутает кодировки.

---

## Стандартизация

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- `StandardScaler()` — для каждого столбца считает $\mu$ и $\sigma$, потом применяет $x_{\text{new}} = (x - \mu) / \sigma$
- `fit_transform(X)` — за один вызов подгоняет (находит μ, σ) и преобразует
- `X_scaled` — это numpy-массив (не DataFrame!) формы `(25192, 41)`

После: каждый столбец имеет среднее 0 и стандартное отклонение 1.

---

## PCA — снижение размерности

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

**Построчно:**
- `PCA(n_components=2)` — создаём PCA с целью получить 2 компоненты
- `pca.fit_transform(X_scaled)`:
  - `fit` — считает ковариационную матрицу, находит собственные векторы (главные компоненты)
  - `transform` — проецирует данные на 2 главных направления
- `X_pca` — массив формы `(25192, 2)`. Каждая строка = одна сетевая сессия в координатах (PC1, PC2)

---

## Создание DataFrame с PCA-координатами

```python
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df.head()
```

- `pd.DataFrame(array, columns=[...])` — оборачивает numpy-массив в DataFrame с именами столбцов
- Это нужно для удобной работы с seaborn (он любит DataFrame)

---

## Explained variance ratio

```python
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print()
print("Total explained variance:", pca.explained_variance_ratio_.sum())
```

- `pca.explained_variance_ratio_` — массив `[0.198, 0.133]` — доля дисперсии каждой компоненты
- `.sum()` → 0.331 — суммарная доля сохранённой дисперсии

Это атрибут с подчёркиванием в конце (`_`) — соглашение sklearn для "вычисленных при fit" атрибутов.

---

## Визуализация PCA с реальными метками

```python
plt.figure(figsize=(10, 6))

pca_df["real_label"] = target.values
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="real_label",
    s=12,
    alpha=0.7
)

plt.title("PCA Projection of Network Traffic (Real Labels)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()
```

**Построчно:**
- `plt.figure(figsize=(10, 6))` — создаём фигуру 10×6 дюймов
- `pca_df["real_label"] = target.values` — добавляем столбец с реальными метками. `.values` нужен чтобы избежать проблем с индексами
- `sns.scatterplot(...)` — точечная диаграмма seaborn
  - `data=pca_df` — источник
  - `x, y` — какие столбцы взять для осей
  - `hue="real_label"` — раскрасить точки по этому столбцу (разные цвета для `normal` и `anomaly`)
  - `s=12` — размер точки
  - `alpha=0.7` — прозрачность (чтобы видеть наложения)
- `plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")` — легенду в правый верхний угол ВНЕ графика (чтобы не закрывала точки)

---

## HDBSCAN — кластеризация

```python
hdb = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
cluster_labels = hdb.fit_predict(X_pca)
```

**Построчно:**
- `hdbscan.HDBSCAN(...)` — создаём модель
  - `min_cluster_size=50` — кластер должен содержать минимум 50 точек, иначе считается шумом
  - `min_samples=10` — насколько плотным должен быть кластер (фактически — параметр core distance)
- `fit_predict(X_pca)` — подгоняет модель на 2D-данных и возвращает массив меток кластеров
- `cluster_labels` — массив длины 25192, значения: `0, 1, 2, ...` (номер кластера) или `-1` (шум/аномалия)

---

## Уникальные метки кластеров

```python
np.unique(cluster_labels)
```

Показывает какие метки получились. Например `array([-1, 0, 1, 2, 3])` означает: 4 кластера + аномалии.

```python
n_anomalies = np.sum(cluster_labels == -1)
print("Detected anomalies:", n_anomalies)
```

- `cluster_labels == -1` — массив True/False
- `np.sum(...)` — сумма True (True = 1) = количество аномалий

---

## Сохранение результатов в DataFrame

```python
pca_df["cluster"] = cluster_labels
pca_df["is_anomaly"] = pca_df["cluster"] == -1
pca_df.head()
```

- Добавляем столбец с номером кластера
- `is_anomaly` — булев столбец: True для аномалий (полезно для фильтрации)

---

## Визуализация HDBSCAN

```python
plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="tab20",
    s=15,
    linewidth=0
)

plt.title("HDBSCAN Clustering Results")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()
```

- `hue="cluster"` — раскрасить по номеру кластера
- `palette="tab20"` — палитра из 20 различимых цветов (хватит на много кластеров)
- `linewidth=0` — точки без обводки (быстрее рендерится для 25k точек)

Аномалии (`cluster=-1`) получат свой отдельный цвет в легенде.

---

## KMeans — кластеризация

```python
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
```

**Построчно:**
- `KMeans(...)`:
  - `n_clusters=2` — заранее заданное число кластеров (нормальный/аномальный)
  - `random_state=42` — фиксирует случайный seed → воспроизводимый результат
  - `n_init=10` — алгоритм запускается 10 раз с разными начальными центроидами, берётся лучший по inertia (сумме квадратов расстояний до центроидов). Защита от попадания в локальный минимум
- `fit_predict(X_scaled)` — обучает и возвращает метки. **Важно:** KMeans запускается на исходных стандартизованных данных (41 признак), а не на X_pca

Почему на `X_scaled`, а не на `X_pca`? Так задано в задании. KMeans в высокой размерности работает (в отличие от плотностных методов), и используя все 41 признак мы даём ему максимум информации.

---

## Визуализация KMeans

```python
pca_df["kmeans_cluster"] = kmeans_labels

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="kmeans_cluster",
    palette="tab10",
    s=12,
    alpha=0.7
)
plt.title("K-Means Clustering on Network Traffic")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()
```

**Тонкость:** KMeans построил кластеры в 41-мерном пространстве, но мы визуализируем их в PCA-проекции. Поэтому граница между кластерами не выглядит как прямая линия в 2D — это проекция гиперплоскости из 41D.

- `palette="tab10"` — 10 цветов (нам нужно 2, но запас не мешает)

---

## Локальный запуск

Всё уже подготовлено:
- `Train_data.csv` и `Test_data.csv` лежат рядом с ноутбуком.
- Путь в первой ячейке: `pd.read_csv("Train_data.csv")`.
- `hdbscan==0.8.42` установлен в `.venv`.

Запустить:
```bash
source .venv/bin/activate
cd "Task 6 NARXOZ AML: Clustering+PCA+AnomalyDet"
jupyter nbconvert --to notebook --execute --inplace narxoz-aml-clustering-pca-anomalydet.ipynb
```

Прогон занимает около минуты.

## Фактические числа прогона

- **Explained variance ratio:** `[0.19786294, 0.13288356]`, сумма ≈ **0.3307**.
- **HDBSCAN:** 39 кластеров + 8019 точек как `-1` (≈31.8%).
- **HDBSCAN vs ground truth:** из 8019 точек `-1` только 1842 реально аномальны → precision ≈ 23%, recall на аномалиях ≈ 16%.
- **KMeans vs ground truth:**

| Кластер | anomaly | normal |
|---|---|---|
| 0 | 2272 | 13327 |
| 1 | 9471 | 122 |

KMeans precision на cluster 1 (аномалия) = **98.7%**, recall = **80.6%**. На этом датасете KMeans сильно обгоняет HDBSCAN — подробнее в `explanation.md` (Q4, Q5).
