# Разбор кода: kNN

---

## Часть 1 — `k_nearest_neighbor.py`

Это главный файл с реализацией алгоритма. Содержит один класс `KNearestNeighbor` с пятью методами.

---

### Импорты

```python
import numpy as np
import math
```

- `numpy` — библиотека для работы с массивами и матрицами. Почти всё в ML делается через неё.
- `math` — стандартная библиотека Python. Используется только в `compute_distances_two_loops` для `math.sqrt()`.

---

### Класс `KNearestNeighbor`

```python
class KNearestNeighbor:
```

Класс — это шаблон объекта. Создаём один раз, потом вызываем методы. В sklearn все алгоритмы устроены так же: `fit()` → `predict()`.

---

### Метод `fit`

```python
def fit(self, X, y):
    self.X_train = X
    self.y_train = y
```

**Что делает:** сохраняет тренировочные данные внутри объекта.

**Построчно:**
- `def fit(self, X, y)` — принимает матрицу признаков `X` и вектор меток `y`
- `self.X_train = X` — сохраняем матрицу изображений, форма `(1697, 64)`
- `self.y_train = y` — сохраняем метки (какая цифра на каждом изображении), форма `(1697,)`

`self` — это сам объект. Через `self.` мы сохраняем данные так, чтобы другие методы класса могли их использовать.

**Почему так:** kNN ничего не вычисляет при обучении — просто запоминает данные. Всё вычисление происходит при предсказании.

---

### Метод `compute_distances_two_loops`

```python
def compute_distances_two_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            dists[i, j] = math.sqrt(sum((X[i] - self.X_train[j])**2))
    return dists
```

**Что делает:** строит матрицу расстояний `(100, 1697)` через два вложенных цикла.

**Построчно:**
- `num_test = X.shape[0]` — сколько тестовых примеров (100). `.shape[0]` — количество строк матрицы
- `num_train = self.X_train.shape[0]` — сколько тренировочных примеров (1697)
- `dists = np.zeros((num_test, num_train))` — создаём пустую матрицу 100×1697, заполненную нулями. Сюда будем записывать расстояния
- `for i in range(num_test)` — внешний цикл: перебираем каждый тестовый пример
- `for j in range(num_train)` — внутренний цикл: для каждого теста перебираем каждый трейн
- `X[i] - self.X_train[j]` — вычитаем два вектора по 64 числа, получаем вектор разностей
- `** 2` — возводим каждый элемент в квадрат
- `sum(...)` — суммируем все 64 квадрата (питоновский `sum`)
- `math.sqrt(...)` — берём квадратный корень → получаем евклидово расстояние
- `dists[i, j] = ...` — записываем результат в ячейку матрицы
- `return dists` — возвращаем заполненную матрицу

**Итого:** 100 × 1697 = 169 700 итераций. Медленно, но понятно.

---

### Метод `compute_distances_one_loop`

```python
def compute_distances_one_loop(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))
    return dists
```

**Что делает:** то же самое, но убирает внутренний цикл — для одного теста считает расстояния до всех трейнов сразу.

**Ключевая строка:**
```python
dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))
```

Разберём по шагам что происходит с формами массивов:

| Операция | Форма | Пояснение |
|----------|-------|-----------|
| `X[i]` | `(64,)` | один тестовый пример — вектор из 64 пикселей |
| `self.X_train` | `(1697, 64)` | все тренировочные примеры |
| `X[i] - self.X_train` | `(1697, 64)` | broadcasting: X[i] "растягивается" и вычитается из каждой строки |
| `** 2` | `(1697, 64)` | каждый элемент в квадрат |
| `np.sum(..., axis=1)` | `(1697,)` | суммируем по оси 1 (по признакам), получаем 1697 чисел |
| `np.sqrt(...)` | `(1697,)` | корень из каждого числа — вектор расстояний |
| `dists[i, :]` | `(1697,)` | записываем в i-ю строку матрицы |

**Broadcasting** — умение NumPy автоматически "растягивать" меньший массив чтобы совпал по форме с большим. Вместо Python-цикла — операция выполняется на уровне C внутри NumPy, что намного быстрее.

---

### Метод `compute_distances_no_loops`

```python
def compute_distances_no_loops(self, X):
    test_sq  = np.sum(X ** 2, axis=1, keepdims=True)
    train_sq = np.sum(self.X_train ** 2, axis=1)
    cross    = X @ self.X_train.T
    dists    = np.sqrt(test_sq - 2 * cross + train_sq)
    return dists
```

**Что делает:** вычисляет всю матрицу расстояний без единого Python-цикла.

**Математическая идея:** раскрываем формулу евклидового расстояния:

$$\|a - b\|^2 = \|a\|^2 - 2 \cdot a^Tb + \|b\|^2$$

Это позволяет посчитать всё через матричные операции.

**Построчно:**

```python
test_sq = np.sum(X ** 2, axis=1, keepdims=True)
```
- `X ** 2` — каждый пиксель в квадрат, форма `(100, 64)`
- `np.sum(..., axis=1)` — суммируем по признакам, получаем `||a||²` для каждого теста
- `keepdims=True` — сохраняем размерность: форма `(100, 1)` вместо `(100,)`. Важно для broadcasting!

```python
train_sq = np.sum(self.X_train ** 2, axis=1)
```
- Аналогично для трейна, форма `(1697,)` — без keepdims, это строка

```python
cross = X @ self.X_train.T
```
- `@` — оператор матричного умножения
- `X` форма `(100, 64)`, `X_train.T` форма `(64, 1697)`
- Результат `cross` форма `(100, 1697)` — скалярные произведения всех пар

```python
dists = np.sqrt(test_sq - 2 * cross + train_sq)
```
- `test_sq` форма `(100, 1)` + `train_sq` форма `(1697,)` → broadcasting даёт `(100, 1697)`
- Вычитаем `2 * cross` формы `(100, 1697)` → получаем `||a - b||²`
- `np.sqrt(...)` → финальные расстояния, форма `(100, 1697)`

**Почему так быстро:** матричное умножение `X @ X_train.T` реализовано через BLAS — высокооптимизированные математические библиотеки, которые используют все ядра процессора и специальные инструкции CPU.

---

### Метод `predict_labels`

```python
def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = self.y_train[np.argsort(dists[i])[:k]]
        y_pred[i] = np.bincount(closest_y.astype(int)).argmax()
    return y_pred
```

**Что делает:** по матрице расстояний определяет класс каждого тестового примера.

**Построчно:**
- `num_test = dists.shape[0]` — количество тестовых примеров (100)
- `y_pred = np.zeros(num_test)` — массив куда запишем предсказания
- `np.argsort(dists[i])` — сортирует индексы трейн-примеров по возрастанию расстояния от i-го теста. Например `[42, 7, 831, ...]` — сначала индекс ближайшего соседа
- `[:k]` — берём первые k индексов (k ближайших соседей)
- `self.y_train[...]` — получаем метки этих k соседей. Например `[3, 3, 5]`
- `closest_y.astype(int)` — конвертируем в целые числа (нужно для bincount)
- `np.bincount([3, 3, 5])` → `[0, 0, 0, 2, 0, 1]` — считает сколько раз встречается каждое число
- `.argmax()` → `3` — индекс максимального элемента = самая частая метка
- `y_pred[i] = ...` — записываем предсказание

---

### Метод `predict`

```python
def predict(self, X, k=1):
    dists = self.compute_distances_no_loops(X)
    return self.predict_labels(dists, k=k)
```

**Что делает:** обёртка — объединяет вычисление расстояний и предсказание меток в один вызов.

Использует `no_loops` — самую быструю реализацию.

---

## Часть 2 — `knn_assignment_0_01.ipynb`

Ноутбук идёт сверху вниз: загрузка данных → визуализация → тестирование реализации → сравнение с sklearn → замер скорости.

---

### Загрузка датасета

```python
from sklearn import datasets
dataset = datasets.load_digits()
```

`load_digits()` — встроенный датасет sklearn. Возвращает объект с полями:
- `dataset.data` — матрица `(1797, 64)`, каждая строка = одно изображение
- `dataset.target` — вектор `(1797,)`, метки (цифры 0-9)
- `dataset.DESCR` — текстовое описание датасета

---

### Разбивка на трейн и тест

```python
test_border = 100
X_train, y_train = dataset.data[test_border:], dataset.target[test_border:]
X_test,  y_test  = dataset.data[:test_border],  dataset.target[:test_border]
```

- Первые 100 примеров → тест, остальные 1697 → трейн
- `dataset.data[100:]` — срез массива с индекса 100 до конца
- `dataset.data[:100]` — срез с начала до индекса 100
- `num_test = X_test.shape[0]` — сохраняем 100, используется позже при подсчёте accuracy

---

### Визуализация изображений

```python
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].reshape((8, 8)).astype('uint8'))
```

- `np.flatnonzero(y_train == y)` — находит все индексы где метка равна `y`
- `np.random.choice(..., replace=False)` — случайно выбирает без повторений
- `X_train[idx].reshape((8, 8))` — разворачивает вектор из 64 чисел обратно в матрицу 8×8 для отображения

---

### Импорт и создание классификатора

```python
try:
    del KNearestNeighbor
except:
    pass

from k_nearest_neighbor import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.fit(X_train, y_train)
```

- `del KNearestNeighbor` — удаляем старую версию класса из памяти (грязный хак для autoreload)
- `from k_nearest_neighbor import ...` — импортируем наш класс из файла `k_nearest_neighbor.py`
- `classifier.fit(X_train, y_train)` — "обучаем" (на самом деле просто запоминаем данные)

---

### Тест двойного цикла

```python
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)  # (100, 1697)
```

Проверяем что метод работает и возвращает матрицу правильной формы.

---

### Визуализация матрицы расстояний

```python
plt.imshow(dists, interpolation='none')
plt.show()
```

Отображает матрицу 100×1697 как изображение. Тёмные ячейки = маленькое расстояние (похожие изображения), светлые = большое (непохожие).

---

### Предсказание с k=1 и k=5

```python
y_test_pred = classifier.predict_labels(dists, k=1)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
```

- `y_test_pred == y_test` — поэлементное сравнение, возвращает массив True/False
- `np.sum(...)` — суммирует True как 1, False как 0 → количество правильных ответов
- `float(num_correct) / num_test` — доля правильных от 0 до 1

---

### Проверка одного цикла и no_loops

```python
dists_one = classifier.compute_distances_one_loop(X_test)
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('One loop difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
```

- `np.linalg.norm(..., ord='fro')` — норма Фробениуса: квадратный корень из суммы квадратов всех элементов разности матриц. Это "расстояние" между двумя матрицами
- Если разница < 0.001 — матрицы считаются одинаковыми (небольшая погрешность допустима из-за особенностей вычислений с плавающей точкой)

---

### Сравнение с sklearn

```python
external_knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
external_knn.fit(X_train, y_train)

y_predicted = implemented_knn.predict(X_test, k=n_neighbors).astype(int)

assert np.array_equal(
    external_knn.predict(X_test),
    y_predicted
), 'Labels predicted by handcrafted and sklearn kNN implementations are different!'
```

- `assert` — если условие ложно, выбрасывает ошибку с сообщением
- `np.array_equal(a, b)` — True если все элементы массивов одинаковы
- Проверяем что наша реализация даёт точно такие же предсказания как библиотечная

---

### Замер скорости

```python
def time_function(f, *args):
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier_big.compute_distances_two_loops, X_test_big)
```

- `time.time()` — возвращает текущее время в секундах
- `toc - tic` — разница = время выполнения функции
- `*args` — передаём аргументы функции через распаковку
- Для более заметной разницы данные увеличены в 5 раз: `np.vstack([X_train]*5)` — вертикально стыкуем матрицу с собой 5 раз

**Результаты на нашем датасете (5× увеличенном):**

| Реализация | Время |
|-----------|-------|
| Двойной цикл | ~20 сек |
| Один цикл | ~0.66 сек |
| Без циклов | ~0.028 сек |

Без циклов быстрее двойного цикла в **~700 раз**.
