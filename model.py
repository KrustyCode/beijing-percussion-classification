"""Classification models written from scratch with NumPy.

Every model follows the same interface as scikit-learn:

    model = KNN(k=5, distance="euclidean")      # or SVM(C=10, kernel="rbf")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

`X` is a feature matrix of shape ``(n_samples, n_features)`` and `y` holds
one label per sample (strings such as ``"daluo"`` or integers both work).
Scale the features before fitting (see `preprocessing.fit_scaler` and
`preprocessing.z_score_standardization`): KNN distances and the SVM's RBF
and polynomial kernels otherwise let the features with the largest values
dominate.

Models:
    KNN        k-nearest-neighbours, any number of classes.
    BinarySVM  Support vector machine for exactly two classes, trained with
               SMO (Sequential Minimal Optimization).
    SVM        Any number of classes: one BinarySVM per pair of classes
               (one-vs-one) and a majority vote, like scikit-learn's ``SVC``.

Trained models can be stored with `save_model` and read back with
`load_model`.
"""
import pickle
import random
import warnings
from abc import ABC, abstractmethod
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Literal, Tuple

import numpy as np

from config import MODEL_FOLDER_PATH


class NotTrainedError(RuntimeError):
    """Raised when predict() is called before fit()."""
    def __init__(self, message="Model is not trained yet. Call fit() before predict()."):
        super().__init__(message)


class SaveModelError(Exception):
    """Raised when a model cannot be written to disk."""
    def __init__(self, message="Model cannot be saved"):
        super().__init__(message)


class ConvergenceWarning(UserWarning):
    """Warned when SVM training stops at `max_iteration` before converging."""


def _check_fit_input(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """Convert `X`, `y` to arrays and check their shapes match.

    Raises:
        ValueError: If `X` is not 2-D or `X` and `y` differ in length.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"'X' must be 2-D (n_samples, n_features), got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"'X' has {len(X)} samples but 'y' has {len(y)} labels")
    return X, y


def _check_predict_input(X, n_features: int) -> np.ndarray:
    """Convert `X` to a 2-D array (one sample becomes one row) and check its width.

    Raises:
        ValueError: If `X` does not have `n_features` features.
    """
    X = np.atleast_2d(np.asarray(X))
    if X.shape[1] != n_features:
        raise ValueError(f"'X' has {X.shape[1]} features but the model was "
                         f"fitted with {n_features}")
    return X


class Model(ABC):
    """Base class for all classifiers: `fit` on training data, then `predict`."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        """Learn from features `X` of shape ``(n_samples, n_features)`` and labels `y`."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return one predicted label per row of `X`."""
        pass


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

class KNN(Model):
    """k-nearest-neighbours classifier.

    `fit` only stores the training data. `predict` finds the `k` training
    samples closest to each query and returns the label most of them share.

    Ties in the vote go to the label that sorts first (alphabetically for
    strings, smallest for numbers), the same as scikit-learn's
    ``KNeighborsClassifier``. An odd `k` avoids ties between two classes.

    Args:
        k: Number of neighbours that vote. Must be at least 1.
        distance: ``"euclidean"`` (straight-line, ``sqrt(sum((a - b)**2))``)
            or ``"manhattan"`` (``sum(|a - b|)``).

    Attributes:
        X: Training features, set by `fit`.
        y: Training labels, set by `fit`.

    Raises:
        ValueError: If `k` is less than 1 or `distance` is unknown.
    """

    def __init__(self, k: int, distance: str = "euclidean"):
        if k < 1:
            raise ValueError(f"'k'({k}) must be at least 1")

        self.k = k
        self.functions = {
            "manhattan": self._manhattan,
            "euclidean": self._euclidian
        }
        if distance not in self.functions:
            raise ValueError(f"'distance'({distance}) must be one of {list(self.functions)}")

        self.distance = distance
        self.selected_distance = self.functions[distance]
        self.X = None
        self.y = None

    def _manhattan(self, a: np.ndarray, b: np.ndarray) -> np.float32:
        """Manhattan (L1) distance: sum of absolute differences."""
        return np.sum(np.abs(a - b))

    def _euclidian(self, a: np.ndarray, b: np.ndarray) -> np.float32:
        """Euclidean (L2) distance: square root of the sum of squared differences."""
        return np.sqrt(np.sum((a - b) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """Store the training data.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
            y: Training labels of shape ``(n_samples,)``. A list works too.

        Returns:
            The model itself, so calls can be chained.

        Raises:
            ValueError: If `X` is not 2-D, `X` and `y` differ in length, or
                there are fewer training samples than `k`.
        """
        X, y = _check_fit_input(X, y)
        if len(X) < self.k:
            raise ValueError(f"'k'({self.k}) is larger than the number of training samples ({len(X)})")

        self.X = X
        self.y = y
        print("Model ready to be used for prediction!!!")
        return self

    def _predict_one(self, a: np.ndarray):
        """Predict the label of a single sample `a` of shape ``(n_features,)``.

        Steps:
            1. Measure the distance from `a` to every training sample.
            2. Keep the labels of the `k` closest ones.
            3. Return the label that appears most often among them.
        """
        # STEP 1) Distance to every training sample
        distances = [self.selected_distance(a, data) for data in self.X]

        # STEP 2) Labels of the k nearest neighbours
        sort_indices = np.argsort(distances)[:self.k]
        y_sorted = self.y[sort_indices]

        # STEP 3) Majority vote (np.unique sorts the labels, so ties go to the first)
        values, counts = np.unique(y_sorted, return_counts=True)
        max_count_index = np.argmax(counts)
        return values[max_count_index]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the label of every sample in `X`.

        Args:
            X: Features of shape ``(n_samples, n_features)``, or a single
                sample of shape ``(n_features,)``.

        Returns:
            Array of shape ``(n_samples,)`` with one label per sample (length 1
            for a single sample).

        Raises:
            NotTrainedError: If `fit` has not been called yet.
            ValueError: If `X` has a different number of features than the
                training data.
        """
        if self.X is None:
            raise NotTrainedError()

        X = _check_predict_input(X, self.X.shape[1])
        return np.array([self._predict_one(a) for a in X])


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

Kernel = Literal["linear", "poly", "rbf"]
KERNELS: Tuple[str, ...] = ("linear", "poly", "rbf")

# Training also stops after this many full passes in a row that change no
# alpha: the remaining KKT violations are within reach of no partner j, so
# more passes would loop forever.
_MAX_PASSES_WITHOUT_CHANGE: int = 5

# Alphas above this count as non-zero (the sample is a support vector).
_ALPHA_EPSILON: float = 1e-5


def _check_svm_params(C, kernel, degree, gamma, tolerance, max_iteration) -> None:
    """Validate SVM hyperparameters.

    Raises:
        ValueError: If any value is out of range or unknown.
    """
    if C <= 0:
        raise ValueError(f"'C'({C}) must be greater than 0")
    if kernel not in KERNELS:
        raise ValueError(f"'kernel'({kernel}) must be one of {list(KERNELS)}")
    if int(degree) != degree or degree < 1:
        raise ValueError(f"'degree'({degree}) must be a whole number of at least 1")
    if isinstance(gamma, str):
        if gamma not in ("scale", "auto"):
            raise ValueError(f"'gamma'({gamma}) must be 'scale', 'auto' or a number > 0")
    elif gamma <= 0:
        raise ValueError(f"'gamma'({gamma}) must be 'scale', 'auto' or a number > 0")
    if tolerance <= 0:
        raise ValueError(f"'tolerance'({tolerance}) must be greater than 0")
    if max_iteration == 0 or max_iteration < -1:
        raise ValueError(f"'max_iteration'({max_iteration}) must be -1 (no limit) or at least 1")


class BinarySVM(Model):
    """Support vector machine for exactly two classes, trained with SMO.

    Training solves the soft-margin dual problem: it finds one weight
    (alpha) per training sample, between 0 and `C`. Samples with a
    non-zero alpha are the support vectors, and the prediction for `x` is
    the sign of ``sum(alpha_i * y_i * K(x_i, x)) + bias``.

    SMO repeatedly picks a sample i that breaks the KKT optimality
    conditions by more than `tolerance`, pairs it with a random sample j,
    and solves for the best alpha_i and alpha_j analytically. Training stops
    when no sample breaks the conditions, when `_MAX_PASSES_WITHOUT_CHANGE`
    passes in a row change nothing, or after `max_iteration` passes.

    Use `SVM` for more than two classes.

    Args:
        C: Penalty for misclassified training samples. Larger values fit the
            training data more tightly; smaller values give a wider margin.
        kernel: ``"linear"`` ``(a . b)``, ``"poly"``
            ``(gamma * (a . b) + coef0) ** degree`` or ``"rbf"``
            ``exp(-gamma * ||a - b||**2)``.
        degree: Polynomial degree, used by ``"poly"`` only.
        gamma: Kernel coefficient for ``"poly"`` and ``"rbf"``. ``"scale"``
            uses ``1 / (n_features * X.var())``, ``"auto"`` uses
            ``1 / n_features`` (both computed in `fit`), or pass a number.
        coef0: Constant term, used by ``"poly"`` only.
        random_state: Seed for picking the partner sample j. ``None`` gives
            different (but equally valid) results on every run.
        tolerance: How far the KKT conditions may be broken before a sample
            is optimised again.
        max_iteration: Maximum number of passes over the training data, or
            -1 for no limit.

    Attributes:
        classes: The two labels, sorted. ``classes[1]`` is the positive (+1)
            class inside the model.
        support_vectors: Training samples with a non-zero alpha, shape
            ``(n_support, n_features)``.
        support_y: Their labels as -1 / +1.
        support_alpha: Their alphas.
        bias: Intercept of the decision function.
        n_iteration: Number of passes the last `fit` ran.

    Raises:
        ValueError: If a hyperparameter is out of range or unknown.
    """

    def __init__(
            self, C: float = 1.0,
            kernel: Kernel = 'linear',
            degree: int = 3, gamma: Literal['scale', 'auto'] | float = 'scale',
            coef0: float = 0.0, random_state: int | None = None,
            tolerance: float = 1e-3, max_iteration: int = -1):
        _check_svm_params(C, kernel, degree, gamma, tolerance, max_iteration)
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.kernel_list: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
            'linear': self._linear,
            'poly': self._poly,
            'rbf': self._rbf,
        }

        self.classes: np.ndarray | None = None
        self.support_vectors: np.ndarray | None = None
        self.support_y: np.ndarray | None = None
        self.support_alpha: np.ndarray | None = None
        self.bias: float | None = None
        self.n_iteration: int = 0
        self._gamma: float | None = None   # numeric gamma, resolved in fit

    # Kernels take a matrix of samples on each side, A (n, d) and B (m, d),
    # and return every pairwise value at once as an (n, m) matrix.
    def _linear(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B.T

    def _poly(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (self._gamma * (A @ B.T) + self.coef0) ** self.degree

    def _rbf(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, clipped at 0 against rounding
        sq_dist = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * (A @ B.T)
        return np.exp(-self._gamma * np.maximum(sq_dist, 0.0))

    def _resolve_gamma(self, X: np.ndarray) -> float:
        """Turn ``"scale"`` / ``"auto"`` into a number for the training data `X`."""
        if self.gamma == "scale":
            variance = X.var()
            return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _decision(self, i: int, alpha: np.ndarray, y: np.ndarray, Kmat: np.ndarray, b: float) -> float:
        """f(x_i): the model's raw output for training sample i."""
        return (alpha * y) @ Kmat[:, i] + b

    def _error(self, i: int, alpha: np.ndarray, y: np.ndarray, Kmat: np.ndarray, b: float) -> float:
        """E_i = f(x_i) - y_i: prediction error for training sample i."""
        return self._decision(i, alpha, y, Kmat, b) - y[i]

    def _get_random_index(self, rng: random.Random, i: int, start: int, end: int) -> int:
        """Return a random index in ``[start, end]`` that is not `i`."""
        while True:
            j = rng.randint(start, end)
            if j != i:
                return j

    def _store_train_result(self, alpha: np.ndarray, bias: float, X: np.ndarray, y: np.ndarray) -> None:
        """Keep only the support vectors (alpha > 0) and the bias."""
        is_support = alpha > _ALPHA_EPSILON
        self.support_vectors = X[is_support]
        self.support_y = y[is_support]
        self.support_alpha = alpha[is_support]
        self.bias = bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinarySVM":
        """Train on two-class data with SMO.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
            y: Training labels with exactly two distinct values, any type.

        Returns:
            The model itself, so calls can be chained.

        Raises:
            ValueError: If `X` is not 2-D, `X` and `y` differ in length, or
                `y` does not have exactly two classes.

        Warns:
            ConvergenceWarning: If training stopped at `max_iteration`.
        """
        X, y = _check_fit_input(X, y)
        X = X.astype(np.float64)
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError(f"BinarySVM needs exactly 2 classes, got {len(self.classes)}; "
                             f"use SVM for more")
        y = np.where(y == self.classes[1], 1.0, -1.0)   # SMO needs labels -1 / +1

        rng = random.Random(self.random_state)
        self._gamma = self._resolve_gamma(X)

        m: int = X.shape[0]
        kernel: Callable = self.kernel_list[self.kernel]
        Kmat: np.ndarray = kernel(X, X)

        alpha: np.ndarray = np.zeros(m)
        bias: float = 0.0
        current_iteration: int = 0
        passes_without_change: int = 0

        while True:
            kkt_violation = 0
            num_changed = 0
            for i in range(m):
                E_i = self._error(i, alpha, y, Kmat, bias)
                r_i = y[i] * E_i
                if (r_i < -self.tolerance and alpha[i] < self.C) or (r_i > self.tolerance and alpha[i] > 0):
                    kkt_violation += 1
                    j = self._get_random_index(rng, i, 0, m-1)
                    E_j = self._error(j, alpha, y, Kmat, bias)
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i]); H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C);  H = min(self.C, alpha[i] + alpha[j])
                    if L == H:
                        continue

                    curvature = 2 * Kmat[i, j] - Kmat[i, i] - Kmat[j, j]
                    if curvature >= 0:
                        continue

                    alpha[j] = alpha[j] - y[j] * (E_i - E_j) / curvature
                    alpha[j] = np.clip(alpha[j], L, H)

                    if np.abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
                    bias_1 = bias - E_i - y[i] * (alpha[i] - alpha_i_old) * Kmat[i, i] - y[j] * (alpha[j] - alpha_j_old) * Kmat[i, j]
                    bias_2 = bias - E_j - y[i] * (alpha[i] - alpha_i_old) * Kmat[i, j] - y[j] * (alpha[j] - alpha_j_old) * Kmat[j, j]

                    if 0 < alpha[i] < self.C: bias = bias_1
                    elif 0 < alpha[j] < self.C: bias = bias_2
                    else: bias = (bias_1+bias_2)/2
                    num_changed += 1

            current_iteration += 1

            # Converged: every sample satisfies the KKT conditions
            if kkt_violation == 0:
                break
            # Stuck: violations left, but no pair can improve them any more
            passes_without_change = 0 if num_changed > 0 else passes_without_change + 1
            if passes_without_change >= _MAX_PASSES_WITHOUT_CHANGE:
                break
            # Out of budget
            if self.max_iteration > 0 and current_iteration >= self.max_iteration:
                warnings.warn(f"SVM stopped at max_iteration={self.max_iteration} before "
                              f"converging ({kkt_violation} KKT violations left)", ConvergenceWarning)
                break

        self.n_iteration = current_iteration
        self._store_train_result(alpha, bias, X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw output ``sum(alpha_i * y_i * K(x_i, x)) + bias`` for every row of `X`.

        Positive values mean ``classes[1]``, negative values ``classes[0]``;
        the further from 0, the more confident.

        Raises:
            NotTrainedError: If `fit` has not been called yet.
        """
        if self.bias is None:
            raise NotTrainedError()

        X = _check_predict_input(X, self.support_vectors.shape[1]).astype(np.float64)
        kernel = self.kernel_list[self.kernel]
        return kernel(X, self.support_vectors) @ (self.support_alpha * self.support_y) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the label (one of `classes`) of every row of `X`.

        Args:
            X: Features of shape ``(n_samples, n_features)``, or a single
                sample of shape ``(n_features,)``.

        Returns:
            Array of shape ``(n_samples,)``.

        Raises:
            NotTrainedError: If `fit` has not been called yet.
        """
        return np.where(self.decision_function(X) >= 0, self.classes[1], self.classes[0])


class SVM(Model):
    """Multi-class support vector machine (one-vs-one).

    `fit` trains one `BinarySVM` for every pair of classes, on the samples
    of those two classes only (6 models for 4 classes). `predict` lets every
    pair model vote and returns the class with the most votes; a tie goes to
    the class that sorts first. This is the same scheme scikit-learn's
    ``SVC`` uses. With two classes it is a single `BinarySVM`.

    Args:
        Same as `BinarySVM`; every pair model gets the same values.

    Attributes:
        classes: All labels, sorted.
        models: The trained pair models, keyed by ``(index_a, index_b)``
            into `classes`.

    Raises:
        ValueError: If a hyperparameter is out of range or unknown.
    """

    def __init__(
            self, C: float = 1.0,
            kernel: Kernel = 'linear',
            degree: int = 3, gamma: Literal['scale', 'auto'] | float = 'scale',
            coef0: float = 0.0, random_state: int | None = None,
            tolerance: float = 1e-3, max_iteration: int = -1):
        _check_svm_params(C, kernel, degree, gamma, tolerance, max_iteration)
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state
        self.tolerance = tolerance
        self.max_iteration = max_iteration

        self.classes: np.ndarray | None = None
        self.models: Dict[Tuple[int, int], BinarySVM] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """Train one `BinarySVM` per pair of classes.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
            y: Training labels with at least two distinct values, any type.

        Returns:
            The model itself, so calls can be chained.

        Raises:
            ValueError: If `X` is not 2-D, `X` and `y` differ in length, or
                `y` has fewer than two classes.
        """
        X, y = _check_fit_input(X, y)
        self.classes = np.unique(y)
        if len(self.classes) < 2:
            raise ValueError(f"SVM needs at least 2 classes, got {len(self.classes)}")

        self.models = {}
        for a, b in combinations(range(len(self.classes)), 2):
            in_pair = (y == self.classes[a]) | (y == self.classes[b])
            self.models[(a, b)] = BinarySVM(
                C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                coef0=self.coef0, random_state=self.random_state,
                tolerance=self.tolerance, max_iteration=self.max_iteration,
            ).fit(X[in_pair], y[in_pair])

        print("Model ready to be used for prediction!!!")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the label of every row of `X` by one-vs-one voting.

        Args:
            X: Features of shape ``(n_samples, n_features)``, or a single
                sample of shape ``(n_features,)``.

        Returns:
            Array of shape ``(n_samples,)`` with labels from `classes`.

        Raises:
            NotTrainedError: If `fit` has not been called yet.
        """
        if not self.models:
            raise NotTrainedError()

        X = np.atleast_2d(np.asarray(X))
        votes = np.zeros((X.shape[0], len(self.classes)), dtype=int)
        for (a, b), model in self.models.items():
            # BinarySVM's positive class is the later label, classes[b]
            positive = model.decision_function(X) >= 0
            votes[:, b] += positive
            votes[:, a] += ~positive
        # argmax picks the first class on a tie
        return self.classes[np.argmax(votes, axis=1)]


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------

def save_model(model: Model, file_name: str = "model", folder: Path = MODEL_FOLDER_PATH) -> Path:
    """Save a model to ``folder / f"{file_name}.pkl"`` with pickle.

    The folder is created if needed.

    Returns:
        Path of the saved file.

    Raises:
        SaveModelError: If the file cannot be written; the original error
            is attached as its cause.
    """
    path = Path(folder) / f"{file_name}.pkl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    except (OSError, pickle.PicklingError) as e:
        raise SaveModelError(f"Model cannot be saved to '{path}': {e}") from e
    print(f"model saved to {path}")
    return path


def load_model(file_name: str = "model", folder: Path = MODEL_FOLDER_PATH) -> Model:
    """Load a model saved by `save_model`.

    Only load files you created yourself: unpickling can run arbitrary code.
    """
    with open(Path(folder) / f"{file_name}.pkl", 'rb') as file:
        return pickle.load(file)
