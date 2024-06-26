# -*- coding: utf-8 -*-
"""vk_ml_hw1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zF3wV3NDO0p9Z2DbymiEka3AZWKy9Gd9

# Машинное обучение
## Домашнее задание №1: KNN + Линейные модели

**Срок сдачи:** 5 марта 2023, 23:59

**Максимально баллов:** 10

**Штраф за опоздание:** по 2 балла за 24 часа задержки. Через 5 дней домашнее задание сгорает.

При отправлении ДЗ указывайте фамилию в названии файла. Формат сдачи будет указан чуть позже.

Используйте данный Ipython Notebook при оформлении домашнего задания.

**Штрафные баллы:**

1. Отсутствие фамилии в имени скрипта (скрипт должен называться по аналогии со stroykova_hw1.ipynb) -1 баллов
2. Все строчки должны быть выполнены. Нужно, чтобы output команды можно было увидеть уже в git'е. В противном случае -1 баллов

При оформлении ДЗ нужно пользоваться данным файлом в качестве шаблона. Не нужно удалять и видоизменять написанный код и текст, если явно не указана такая возможность.

## KNN (5 баллов)
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml, fetch_20newsgroups

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.metrics import accuracy_score

from scipy.spatial.distance import cdist
from collections import Counter

"""##### Задание 1 (1 балл)
Реализовать KNN в классе MyKNeighborsClassifier (обязательное условие: точность не ниже sklearn реализации)
Разберитесь самостоятельно, какая мера расстояния используется в KNeighborsClassifier дефолтно и реализуйте свой алгоритм именно с этой мерой.
Для подсчета расстояний можно использовать функции [отсюда](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
"""

class MyKNeighborsClassifier(BaseEstimator):
    # поле метрика добавлено для 4 задания
    def __init__(self, n_neighbors, algorithm='brute', metric='minkowski'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric

    def fit(self, X, y):
        self.y_train = y
        if self.algorithm == 'brute':
            self.X_train = X
        elif self.algorithm == 'kd_tree':
          self.tree = KDTree(X)
        return self

    def predict(self, X):
        y_pred = []

        if self.algorithm == 'brute':
            # если будет метрика миньковского, то p по дефолту будет 2, нам это и нужно
            dist = cdist(X, self.X_train, metric=self.metric)
            for i in range(X.shape[0]):
                k_nearest_indx = np.argsort(dist[i])[:self.n_neighbors]
                k_nearest_class = self.y_train[k_nearest_indx]
                most_common_class = Counter(k_nearest_class).most_common(1)[0][0]
                y_pred.append(most_common_class)

        elif self.algorithm == 'kd_tree':
            dist, indx = self.tree.query(X, k=self.n_neighbors)
            for indx_array in indx:
                k_nearest_class = self.y_train[indx_array]
                most_common_class = Counter(k_nearest_class).most_common(1)[0][0]
                y_pred.append(most_common_class)

        return y_pred

"""**IRIS**

В библиотеке scikit-learn есть несколько датасетов из коробки. Один из них [Ирисы Фишера](https://ru.wikipedia.org/wiki/%D0%98%D1%80%D0%B8%D1%81%D1%8B_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0)
"""

iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, stratify=iris.target)

clf = KNeighborsClassifier(n_neighbors=2, algorithm='brute')
my_clf = MyKNeighborsClassifier(n_neighbors=2, algorithm='brute')

clf.fit(X_train, y_train)
my_clf.fit(X_train, y_train)

sklearn_pred = clf.predict(X_test)
my_clf_pred = my_clf.predict(X_test)
assert abs( accuracy_score(y_test, my_clf_pred) -  accuracy_score(y_test, sklearn_pred ) )<0.005, "Score must be simillar"

"""**Задание 2 (0.5 балла)**

Давайте попробуем добиться скорости работы на fit, predict сравнимой со sklearn для iris. Допускается замедление не более чем в 2 раза.
Для этого используем numpy.
"""

# Commented out IPython magic to ensure Python compatibility.
# %timeit clf.fit(X_train, y_train)

# Commented out IPython magic to ensure Python compatibility.
# %timeit my_clf.fit(X_train, y_train)

# Commented out IPython magic to ensure Python compatibility.
# %timeit clf.predict(X_test)

# Commented out IPython magic to ensure Python compatibility.
# %timeit my_clf.predict(X_test)

"""###### Задание 3 (1 балл)
Добавьте algorithm='kd_tree' в реализацию KNN (использовать KDTree из sklearn.neighbors). Необходимо добиться скорости работы на fit,  predict сравнимой со sklearn для iris. Допускается замедление не более чем в 2 раза.
Для этого используем numpy. Точность не должна уступать значению KNN из sklearn.
"""

clf = KNeighborsClassifier(n_neighbors=2, algorithm='kd_tree')
my_clf = MyKNeighborsClassifier(n_neighbors=2, algorithm='kd_tree')

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, stratify=iris.target)

# Commented out IPython magic to ensure Python compatibility.
# %time clf.fit(X_train, y_train)

# Commented out IPython magic to ensure Python compatibility.
# %time my_clf.fit(X_train, y_train)

# Commented out IPython magic to ensure Python compatibility.
# %time clf.predict(X_test)

# Commented out IPython magic to ensure Python compatibility.
# %time my_clf.predict(X_test)

sklearn_pred = clf.predict(X_test)
my_clf_pred = my_clf.predict(X_test)
assert abs( accuracy_score(y_test, my_clf_pred) -  accuracy_score(y_test, sklearn_pred ) )<0.005, "Score must be simillar"

"""**Задание 4 (2.5 балла)**

Рассмотрим новый датасет 20 newsgroups
"""

newsgroups = fetch_20newsgroups(subset='train', remove=['headers','footers', 'quotes'])

data = newsgroups['data']
target = newsgroups['target']

"""Преобразуйте текстовые данные из data с помощью [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). Словарь можно ограничить по частотности."""

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(lowercase=True, min_df=0.1)
data = vectorizer.fit_transform(data).toarray()

"""*Так мы получили векторное представление наших текстов. Значит можно приступать к задаче обучения модели*

Реализуйте разбиение выборки для кросс-валидации на 3 фолдах. Разрешено использовать sklearn.cross_validation
"""

# функция, которая разобьет выборку размера n_samples на k фолдов
# для удобства возвращать будем индексы
def get_kfold_indicies(n_samples, k=2, shuffle=False, random_seed=None):
  from math import ceil
  if random_seed != None:
    np.random.seed(random_seed)
  indicies = np.arange(0, n_samples, dtype=int)
  if shuffle:
    np.random.shuffle(indicies)
  fold_size = ceil(n_samples / k)
  fold_indicies = []
  for start in range (0, n_samples, fold_size):
    end = min(start + fold_size, n_samples)
    fold_indicies.append(indicies[start:end])
  return fold_indicies

get_kfold_indicies(n_samples=data.shape[0], k=3, shuffle=True, random_seed=42)

"""Напишите метод, позволяющий найти оптимальное количество ближайших соседей(дающее максимальную точность в среднем на валидации на 3 фолдах).
Постройте график зависимости средней точности от количества соседей. Можно рассмотреть число соседей от 1 до 10.
"""

def opt_neighboours(X, y, metric='minkowski'):
  mean_acc_path = []
  fold_indicies = get_kfold_indicies(n_samples=X.shape[0], k=3, shuffle=True, random_seed=42)
  for k in range(1, 11):
    clf = MyKNeighborsClassifier(n_neighbors=k, metric=metric)
    clf.fit(np.concatenate([X[fold_indicies[0]], X[fold_indicies[1]]]),
            np.concatenate([y[fold_indicies[0]], y[fold_indicies[1]]]))
    pred1 = clf.predict(X[fold_indicies[2]])
    acc1 = accuracy_score(y[fold_indicies[2]], pred1)

    clf.fit(np.concatenate([X[fold_indicies[2]], X[fold_indicies[1]]]),
            np.concatenate([y[fold_indicies[2]], y[fold_indicies[1]]]))
    pred2 = clf.predict(X[fold_indicies[0]])
    acc2 = accuracy_score(y[fold_indicies[0]], pred2)

    clf.fit(np.concatenate([X[fold_indicies[0]], X[fold_indicies[2]]]),
            np.concatenate([y[fold_indicies[0]], y[fold_indicies[2]]]))
    pred3 = clf.predict(X[fold_indicies[1]])
    acc3 = accuracy_score(y[fold_indicies[1]], pred3)

    mean_acc = (acc1 + acc2 + acc3) / 3
    mean_acc_path.append(mean_acc)

  return (np.argmax(mean_acc_path) + 1, mean_acc_path)

opt, path = opt_neighboours(data, target)

print(f'оптимальное количество соседей: {opt}')

plt.plot(np.arange(1, 11), path)
plt.title('mean accuracy on 3-folds cv')

plt.show()

"""Как изменится качество на валидации, если:

1. Используется косинусная метрика вместо евклидовой.
2. К текстам применяется TfIdf векторизацию( sklearn.feature_extraction.text.TfidfVectorizer)

Сравните модели, выберите лучшую.
"""

cos_opt, cos_path = opt_neighboours(data, target, metric='cosine')

print(f'оптимальное количество соседей при косинусном расстоянии: {cos_opt}')

plt.plot(np.arange(1, 11), cos_path)
plt.title('mean accuracy on 3-folds cv, cosine metric')

plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(lowercase=True, min_df=0.1)
tfidf_data = tfidf_vec.fit_transform(newsgroups['data']).toarray()

tfidf_opt, tfidf_path = opt_neighboours(tfidf_data, target, metric='minkowski')

print(f"оптимальное колиество соседей: {tfidf_opt}")

plt.plot(np.arange(1, 11), tfidf_path)
plt.title('mean accuracy on 3-folds cv, tfidf vec')

plt.show()

"""Загрузим  теперь test  часть нашей выборки и преобразуем её аналогично с train частью. Не забудьте, что наборы слов в train и test части могут отличаться."""

newsgroups = fetch_20newsgroups(subset='test', remove=['headers','footers', 'quotes'])

X_test = tfidf_vec.transform(newsgroups['data']).toarray()
y_test = newsgroups['target']

best_clf = MyKNeighborsClassifier(n_neighbors=tdifd_opt)
best_clf.fit(tfidf_data, target)
pred = best_clf.predict(X_test)

"""Оценим точность вашей лучшей модели на test части датасета. Отличается ли оно от кросс-валидации? Попробуйте сделать выводы, почему отличается качество."""

print(f'accuracy: {accuracy_score(pred, y_test)}')

"""Точность на кросс-валидации и тесте могут отличаться из-за разницы тестовых и тренировочных данных(как было замечено выше, наборы слов в train и test могут отличаться).

# Линейные модели (5 баллов)
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,5)

"""В этом задании мы будем реализовать линейные модели. Необходимо реализовать линейную и логистическую регрессии с L2 регуляризацией

### Теоретическое введение



Линейная регрессия решает задачу регрессии и оптимизирует функцию потерь MSE

$$L(w) =  \frac{1}{N}\left[\sum_i (y_i - a_i) ^ 2 \right], $$ где $y_i$ $-$ целевая функция,  $a_i = a(x_i) =  \langle\,x_i,w\rangle ,$ $-$ предсказание алгоритма на объекте $x_i$, $w$ $-$ вектор весов (размерности $D$), $x_i$ $-$ вектор признаков (такой же размерности $D$).

Не забываем, что здесь и далее  мы считаем, что в $x_i$ есть тождественный вектор единиц, ему соответствует вес $w_0$.


Логистическая регрессия является линейным классификатором, который оптимизирует так называемый функционал log loss:

$$L(w) = - \frac{1}{N}\left[\sum_i y_i \log a_i + ( 1 - y_i) \log (1 - a_i) \right],$$
где  $y_i  \in \{0,1\}$ $-$ метка класса, $a_i$ $-$ предсказание алгоритма на объекте $x_i$. Модель пытается предсказать апостериорую вероятность объекта принадлежать к классу "1":
$$ p(y_i = 1 | x_i) = a(x_i) =  \sigma( \langle\,x_i,w\rangle ),$$
$w$ $-$ вектор весов (размерности $D$), $x_i$ $-$ вектор признаков (такой же размерности $D$).

Функция $\sigma(x)$ $-$ нелинейная функция, пероводящее скалярное произведение объекта на веса в число $\in (0,1)$ (мы же моделируем вероятность все-таки!)

$$\sigma(x) = \frac{1}{1 + \exp(-x)}$$

Если внимательно посмотреть на функцию потерь, то можно заметить, что в зависимости от правильного ответа алгоритм штрафуется или функцией $-\log a_i$, или функцией $-\log (1 - a_i)$.



Часто для решения проблем, которые так или иначе связаны с проблемой переобучения, в функционал качества добавляют слагаемое, которое называют ***регуляризацией***. Итоговый функционал для линейной регрессии тогда принимает вид:

$$L(w) =  \frac{1}{N}\left[\sum_i (y_i - a_i) ^ 2 \right] + \frac{1}{C}R(w) $$

Для логистической:
$$L(w) = - \frac{1}{N}\left[\sum_i y_i \log a_i + ( 1 - y_i) \log (1 - a_i) \right] +  \frac{1}{C}R(w)$$

Самое понятие регуляризации введено основателем ВМК академиком Тихоновым https://ru.wikipedia.org/wiki/Метод_регуляризации_Тихонова

Идейно методика регуляризации заключается в следующем $-$ мы рассматриваем некорректно поставленную задачу (что это такое можно найти в интернете), для того чтобы сузить набор различных вариантов (лучшие из которых будут являться переобучением ) мы вводим дополнительные ограничения на множество искомых решений. На лекции Вы уже рассмотрели два варианта регуляризации.

$L1$ регуляризация:
$$R(w) = \sum_{j=1}^{D}|w_j|$$
$L2$ регуляризация:
$$R(w) =  \sum_{j=1}^{D}w_j^2$$

С их помощью мы ограничиваем модель в  возможности выбора каких угодно весов минимизирующих наш лосс, модель уже не сможет подстроиться под данные как ей угодно.

Вам нужно добавить соотвествущую Вашему варианту $L2$ регуляризацию.

И так, мы поняли, какую функцию ошибки будем минимизировать, разобрались, как получить предсказания по объекту и обученным весам. Осталось разобраться, как получить оптимальные веса. Для этого нужно выбрать какой-то метод оптимизации.



Градиентный спуск является самым популярным алгоритмом обучения линейных моделей. В этом задании Вам предложат реализовать стохастический градиентный спуск или  мини-батч градиентный спуск (мини-батч на русский язык довольно сложно перевести, многие переводят это как "пакетный", но мне не кажется этот перевод удачным). Далее нам потребуется определение **эпохи**.
Эпохой в SGD и MB-GD называется один проход по **всем** объектам в обучающей выборки.
* В SGD градиент расчитывается по одному случайному объекту. Сам алгоритм выглядит примерно так:
        1) Перемешать выборку
        2) Посчитать градиент функции потерь на одном объекте (далее один объект тоже будем называть батчем)
        3) Сделать шаг спуска
        4) Повторять 2) и 3) пока не пройдет максимальное число эпох.
* В Mini Batch SGD - по подвыборке объектов. Сам алгоритм выглядит примерно так::
        1) Перемешать выборку, выбрать размер мини-батча (от 1 до размера выборки)
        2) Почитать градиент функции потерь по мини-батчу (не забыть поделить на  число объектов в мини-батче)
        3) Сделать шаг спуска
        4) Повторять 2) и 3) пока не пройдет максимальное число эпох.
* Для отладки алгоритма реализуйте возможность  вывода средней ошибки на обучении модели по объектам (мини-батчам). После шага градиентного спуска посчитайте значение ошибки на объекте (или мини-батче), а затем усредните, например, по ста шагам. Если обучение проходит корректно, то мы должны увидеть, что каждые 100 шагов функция потерь уменьшается.
* Правило останова - максимальное количество эпох

## Зачем нужны батчи?


Как Вы могли заметить из теоретического введения, что в случае SGD, что в случа mini-batch GD,  на каждой итерации обновление весов  происходит только по небольшой части данных (1 пример в случае SGD, batch примеров в случае mini-batch). То есть для каждой итерации нам **не нужна вся выборка**. Мы можем просто итерироваться по выборке, беря батч нужного размера (далее 1 объект тоже будем называть батчом).

Легко заметить, что в этом случае нам не нужно загружать все данные в оперативную память, достаточно просто считать батч с диска, обновить веса, считать диска другой батч и так далее. В целях упрощения домашней работы, прямо с диска  мы считывать не будем, будем работать с обычными numpy array.





## Немножко про генераторы в Python



Идея считывания данных кусками удачно ложится на так называемые ***генераторы*** из языка Python. В данной работе Вам предлагается не только разобраться с логистической регрессией, но  и познакомиться с таким важным элементом языка.  При желании Вы можете убрать весь код, связанный с генераторами, и реализовать логистическую регрессию и без них, ***штрафоваться это никак не будет***. Главное, чтобы сама модель была реализована правильно, и все пункты были выполнены.

Подробнее можно почитать вот тут https://anandology.com/python-practice-book/iterators.html


К генератору стоит относиться просто как к функции, которая порождает не один объект, а целую последовательность объектов. Новое значение из последовательности генерируется с помощью ключевого слова ***yield***.

Концепция крайне удобная для обучения  моделей $-$ у Вас есть некий источник данных, который Вам выдает их кусками, и Вам совершенно все равно откуда он их берет. Под ним может скрывать как массив в оперативной памяти, как файл на жестком диске, так и SQL база данных. Вы сами данные никуда не сохраняете, оперативную память экономите.

Если Вам понравилась идея с генераторами, то Вы можете реализовать свой, используя прототип batch_generator. В нем Вам нужно выдавать батчи признаков и ответов для каждой новой итерации спуска. Если не понравилась идея, то можете реализовывать SGD или mini-batch GD без генераторов.
"""

def batch_generator(X, y, shuffle=True, batch_size=1):
    """
    Гератор новых батчей для обучения
    X          - матрица объекты-признаки
    y_batch    - вектор ответов
    shuffle    - нужно ли случайно перемешивать выборку
    batch_size - размер батча ( 1 это SGD, > 1 mini-batch GD)
    Генерирует подвыборку для итерации спуска (X_batch, y_batch)
    """
    from math import ceil

    num_samples = X.shape[0]

    indx = np.arange(num_samples)
    if shuffle:
      np.random.shuffle(indx)

    start = 0
    end = batch_size

    for i in range(ceil(num_samples/batch_size)):
        X_batch = X[indx[start:min(end, num_samples)]]
        y_batch = y[indx[start:min(end, num_samples)]]
        yield (X_batch, y_batch)
        start += batch_size
        end += batch_size

#%%pycodestyle

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


from sklearn.base import BaseEstimator, ClassifierMixin

class MySGDClassifier(BaseEstimator, ClassifierMixin):
    # я добавила возможность задать batch_size при инициализации классификатора
    def __init__(self, batch_generator, batch_size=10, C=1, alpha=0.01, max_epoch=10, model_type='lin_reg'):
        """
        batch_generator -- функция генератор, которой будем создавать батчи
        C - коэф. регуляризации
        alpha - скорость спуска
        max_epoch - максимальное количество эпох
        model_type - тип модели, lin_reg или log_reg
        """

        self.C = C
        self.alpha = alpha
        self.max_epoch = max_epoch
        self.batch_generator = batch_generator
        self.errors_log = {'iter' : [], 'loss' : []}
        self.model_type = model_type
        self.batch_size = batch_size

    def calc_loss(self, X_batch, y_batch):
        """
        Считаем функцию потерь по батчу
        X_batch - матрица объекты-признаки по батчу
        y_batch - вектор ответов по батчу
        Не забудте тип модели (линейная или логистическая регрессия)!
        """
        if self.model_type == 'lin_reg':
          loss = np.longdouble(np.square(X_batch.dot(self.weights) - y_batch).mean() + 1 / self.C * np.sum(np.square(self.weights)))
        elif self.model_type == 'log_reg':
          loss = - np.mean(y_batch * np.log(sigmoid(X_batch.dot(self.weights))) + (1 - y_batch) * np.log(1 - sigmoid(X_batch.dot(self.weights)))) + 1 / self.C * np.sum(np.square(self.weights))
        return loss

    def calc_loss_grad(self, X_batch, y_batch):
        """
        Считаем  градиент функции потерь по батчу (то что Вы вывели в задании 1)
        X_batch - матрица объекты-признаки по батчу
        y_batch - вектор ответов по батчу
        Не забудте тип модели (линейная или логистическая регрессия)!
        """
        if self.model_type == 'lin_reg':
          loss_grad = 2 * X_batch.T.dot(X_batch.dot(self.weights) - y_batch) / X_batch.shape[0] + 2 / self.C * self.weights
        elif self.model_type == 'log_reg':
          loss_grad = X_batch.T.dot(sigmoid(X_batch.dot(self.weights)) - y_batch) / X_batch.shape[0] + 2 / self.C * self.weights
        return loss_grad

    def update_weights(self, new_grad):
        self.weights -= self.alpha * new_grad

    def fit(self, X, y):
        '''
        Обучение модели
        X - матрица объекты-признаки
        y - вектор ответов
        '''

        self.weights = np.random.randn(X.shape[1])
        for n in range(0, self.max_epoch):
            new_epoch_generator = self.batch_generator(X, y, batch_size=self.batch_size)
            for batch_num, new_batch in enumerate(new_epoch_generator):
                X_batch = new_batch[0]
                y_batch = new_batch[1]
                batch_grad = self.calc_loss_grad(X_batch, y_batch)
                self.update_weights(batch_grad)
                # считаем ошибку после шага градиентного спуска,
                # т.к. это помогает отследить эффективность последнего изменения весов
                #print(f"epoch: {n}, iter: {batch_num}")
                batch_loss = self.calc_loss(X_batch, y_batch)
                self.errors_log['iter'].append(batch_num)
                self.errors_log['loss'].append(batch_loss)

        return self

    def predict(self, X):
        '''
        Предсказание класса
        X - матрица объекты-признаки
        Не забудте тип модели (линейная или логистическая регрессия)!
        '''
        if self.model_type == 'lin_reg':
          y_hat = X.dot(self.weights) > 0.5
        elif self.model_type == 'log_reg':
          y_hat = sigmoid(X.dot(self.weights)) >= 0.6
        return y_hat

"""Запустите обе регрессии на синтетических данных.


Выведите полученные веса и нарисуйте разделяющую границу между классами (используйте только первых два веса для первых двух признаков X[:,0], X[:,1] для отображения в 2d пространство ).  
"""

def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, colors=['purple', 'yellow'])
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

np.random.seed(0)

C1 = np.array([[0., -0.8], [1.5, 0.8]])
C2 = np.array([[1., -0.7], [2., 0.7]])
gauss1 = np.dot(np.random.randn(200, 2) + np.array([5, 3]), C1)
gauss2 = np.dot(np.random.randn(200, 2) + np.array([1.5, 0]), C2)

X = np.vstack([gauss1, gauss2])
y = np.r_[np.ones(200), np.zeros(200)]

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

linreg = MySGDClassifier(batch_generator=batch_generator, model_type='lin_reg')
linreg.fit(X, y)

logreg = MySGDClassifier(batch_generator=batch_generator, model_type='log_reg')
logreg.fit(X, y)

print(f'linear regression weights: {linreg.weights}\nlogistic regression weights: {logreg.weights}')

plt.subplot(1, 2, 1)
plt.title('Linear regression Decision boundary')
plot_decision_boundary(linreg, X, y)

plt.subplot(1, 2, 2)
plt.title('Logistic regression Decision boundary')
plot_decision_boundary(logreg, X, y)

"""Далее будем анализировать Ваш алгоритм.
Для этих заданий используйте датасет ниже.
"""

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100000, n_features=10,
                           n_informative=4, n_redundant=0,
                           random_state=123, class_sep=1.0,
                           n_clusters_per_class=1)

"""Покажите сходимости обеих регрессией на этом датасете: изобразите график  функции потерь, усредненной по $N$ шагам градиентого спуска, для разных `alpha` (размеров шага). Разные `alpha` расположите на одном графике.

$N$ можно брать 10, 50, 100 и т.д.

Что Вы можете сказать про сходимость метода при различных `alpha`? Какое значение стоит выбирать для лучшей сходимости?
"""

def get_losses(alpha, model_type, n=10):
  clf = MySGDClassifier(batch_generator=batch_generator, batch_size=32, alpha=alpha, model_type=model_type)
  clf.fit(X, y)
  path = []
  for start in range(0, len(clf.errors_log['loss']), n):
    end = min(start + n, len(clf.errors_log['loss']))
    loss = np.mean(clf.errors_log['loss'][start:end])
    path.append(loss)
  return path

linreg_loss = dict()
for alpha in [0.1, 0.07, 0.04, 0.01, 0.005, 0.001, 1e-4, 1e-5]:
    linreg_loss[alpha] = get_losses(alpha, 'lin_reg', 50)
    plt.plot(linreg_loss[alpha], label=alpha)

plt.legend()
plt.title('linear regression loss')
plt.show()

"""Масштаб у графика получился сомнительный, но даже так заметно, что при уменьшении шага сходимость градиентного спуска становится хуже и хуже(при шаге=0.00001 мы даже на последней итерации не около нуля).

Посмотрим на наибольшие из рассмотренных alpha поближе:
"""

plt.plot(linreg_loss[0.1], label=0.1)
plt.plot(linreg_loss[0.07], label=0.07)
plt.plot(linreg_loss[0.04], label=0.04)
plt.legend()
plt.show()

"""даже по этому графику заметно, что с уменьшением шага, сходимость понемногу ухудшается, поэтому я бы оставила alpha = 0.1."""

logreg_loss = dict()
for alpha in [0.1, 0.07, 0.04, 0.01, 0.005, 0.001, 1e-4, 1e-5]:
    logreg_loss[alpha] = get_losses(alpha, 'log_reg', 50)
    plt.plot(logreg_loss[alpha], label=alpha)

plt.legend()
plt.title('logistic regression loss')
plt.show()

plt.plot(logreg_loss[0.1], label=0.1)
plt.plot(logreg_loss[0.07], label=0.07)
#plt.plot(logreg_loss[0.04], label=0.04)
plt.legend()
plt.show()

"""Похожая ситуация с логистической регрессией, здесь можно выбрать шаг между 0.1 и 0.07.

Изобразите график среднего значения весов для обеих регрессий в зависимости от коеф. регуляризации С из `np.logspace(3, -3, 10)`
"""

linreg_weights = []
logreg_weights = []
for C in np.logspace(3, -3, 10):
  linreg = MySGDClassifier(batch_generator=batch_generator, C=C, batch_size=50, alpha=0.1, model_type='lin_reg')
  linreg.fit(X, y)
  linreg_weights.append(linreg.weights)

  logreg = MySGDClassifier(batch_generator=batch_generator, C=C, batch_size=50, alpha=0.1, model_type='log_reg')
  logreg.fit(X, y)
  logreg_weights.append(logreg.weights)

"""при С <= 0.1 у нас начинает происходить переполнение и вместо весов мы получаем nan'ы(попробовала положить результаты вычислений в длинные типы данных, но не помогло)"""

linreg_weights

logreg_weights

"""Довольны ли Вы, насколько сильно уменьшились Ваши веса?"""

plt.plot(np.mean(logreg_weights, axis=1)[:-4], label='logreg weights')
plt.plot(np.mean(linreg_weights, axis=1)[:-4], label='linreg weights')

plt.legend()
plt.show()

"""на этом графике все достаточно позитивно и веса и вправду уменьшаются по модулю"""

plt.plot(np.mean(logreg_weights, axis=1)[:-3], label='logreg weights')

plt.legend()
plt.show()

"""но почему здесь на последней модели вырастают огромные веса для меня загадка..."""

