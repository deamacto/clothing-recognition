import numpy as np


def get_estimators(image, edge_value):
    return (image >= edge_value).astype(int).flatten()


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    error = 0.0
    for index in range(np.shape(p_y_x)[0]):
        maxVal = 0.0
        maxIndex = -1
        for (labelNo, label) in enumerate(p_y_x[index]):
            if label >= maxVal:
                maxVal = label
                maxIndex = labelNo
        if maxIndex != y_true[index]:
            error += 1
    return error / np.shape(p_y_x)[0]


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    probabilities = np.zeros(10)
    for label in y_train:
        probabilities[label] += 1
    return probabilities / np.shape(y_train)[0]


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    X_trainArr = X_train
    D = np.shape(X_trainArr)[1]
    probabilities = np.empty((10, D))
    yShaped = np.reshape(y_train, (np.shape(y_train)[0], 1))
    Arr = np.append(X_trainArr, yShaped, axis=1)
    for k in range(10):
        for d in range(D):
            numerator = np.count_nonzero(
                (Arr[:, d] == 1) & (Arr[:, D] == k)) + a - 1
            denominator = np.count_nonzero(Arr[:, D] == k) + a + b - 2
            probabilities[k][d] = numerator / denominator
    return probabilities


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    X_arr = X
    pyx = np.zeros(shape=(X_arr.shape[0], 10))

    for i in range(X_arr.shape[0]):
        for j in range(10):
            pyx[i][j] = np.prod(np.power(p_x_1_y, X_arr[i, :]) * np.power((1 - p_x_1_y), (1 - X_arr[i, :])), axis=1)[j] * p_y[j]
        pyx[i] /= np.sum(pyx[i])
    return pyx


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.

    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    bestError = float("inf")
    bestA = 0.0
    bestB = 0.0
    errors = []
    estimateAPriori = estimate_a_priori_nb(y_train)
    for a in a_values:
        errorsA = []
        for b in b_values:
            probXY = estimate_p_x_y_nb(X_train, y_train, a, b)
            probabilities = p_y_x_nb(estimateAPriori, probXY, X_val)
            classError = classification_error(probabilities, y_val)
            errorsA.append(classError)
            if classError < bestError:
                bestError = classError
                bestA = a
                bestB = b
        errors.append(errorsA)
    return (bestError, bestA, bestB)
