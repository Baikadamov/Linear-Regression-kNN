# KNearestNeighbor — конспект реализации

## Общая идея kNN

kNN — «ленивый» алгоритм: не строит модель при обучении.
- **fit** — просто запоминает тренировочные данные
- **predict** — при каждом предсказании сравнивает тест со всеми трейн-примерами

---

## class KNearestNeighbor

### `fit(self, X, y)`

```python
def fit(self, X, y):
    self.X_train = X
    self.y_train = y
```

- `X` — матрица признаков обучающей выборки, shape `(N, D)` где N — кол-во примеров, D — кол-во признаков
- `y` — вектор меток, shape `(N,)`
- Просто сохраняем данные как атрибуты объекта — никакого обучения нет

---

## Следующий шаг: compute_distances_two_loops

Цель: построить матрицу расстояний `dists` shape `(num_test, num_train)`.
`dists[i, j]` = евклидово расстояние между `X_test[i]` и `X_train[j]`.

**Евклидово расстояние** (без `np.linalg.norm`):
```
d(a, b) = sqrt( sum( (a - b)^2 ) )
```

```python
def compute_distances_two_loops(self, X):
    num_test = X.shape[0]               # кол-во тестовых примеров
    num_train = self.X_train.shape[0]   # кол-во трейн примеров
    dists = np.zeros((num_test, num_train))  # пустая матрица num_test x num_train
    for i in range(num_test):
        for j in range(num_train):
            dists[i, j] = math.sqrt(np.sum((X[i] - self.X_train[j])**2))
            # X[i] - X_train[j] → вектор разностей (64 числа)
            # **2                → поэлементно в квадрат
            # np.sum(...)        → сумма квадратов
            # sqrt(...)          → евклидово расстояние
    return dists
```

Сложность: O(num_test × num_train × D) — очень медленно на больших данных.

---

## Следующий шаг: compute_distances_one_loop

Убираем внутренний цикл по `j`. Вместо того чтобы считать расстояние до одного трейн-примера, считаем сразу до всех трейн-примеров для одного тестового `X[i]`.

```
dists[i, :] = sqrt( sum( (X[i] - X_train)^2, axis=1 ) )
```

- `X[i]` имеет shape `(64,)`, `X_train` имеет shape `(1697, 64)`
- numpy автоматически "растягивает" X[i] по строкам (broadcasting)
- `axis=1` — суммируем по признакам (по столбцам), получаем вектор из 1697 расстояний

