"""Evaluation tools: confusion matrix, metrics report, cross-validation and grid search.

Typical use::

    params, cv_accuracy = grid_search({"k": K}, KNN, X_train, y_train, groups=sources_train)
    model = KNN(**params).fit(X_train, y_train)
    cm, labels = confusion_matrix(model.predict(X_test), y_test)
    evaluation_report(cm, labels)

Choosing the clip duration (``MAX_DURATION`` in config.py) by
cross-validation on the original training clips::

    best, history = search_best_duration(DURATIONS, KNN(k=5), train["original"])

When the training set contains augmented copies, pass `groups` (the
``sources`` of `feature_extraction.Features`) to the cross-validation and
grid search. Every copy then stays in the same fold as its original clip;
otherwise a validation clip's own copy sits in the training folds and the
accuracy comes out far too high (with KNN, close to 100% even on random
labels).

Comparing many trained models on every test set and level::

    results, matrices = evaluate_models(svm_models, testing, "SVM")  # one row per test
    results.to_csv("results.csv", index=False)                       # store
    results_table(results, "SVM")                                    # scenarios x test sets
    show_table("SVM", "macro_f1", results)                           # same, coloured (notebook)
    plot_robustness(results, "SVM")                                  # metric vs level
    show_confusion_matrix(*matrices[("SVM", "original", "noise", 0)])
    show_confusion_grid(matrices, "SVM", "noise", 0)                 # one per training scenario
"""
import contextlib
import inspect
import io
import itertools
from typing import Dict, Iterator, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from preprocessing import Dataset, clip_all
from feature_extraction import extract_all_features
from IPython.display import display

from model import Model


def confusion_matrix(predictions: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count how often each true class was predicted as each class.

    Args:
        predictions: Predicted labels.
        y: True labels, same length as `predictions`.

    Returns:
        ``(cm, labels)``: ``cm[i, j]`` is the number of samples of class
        ``labels[i]`` predicted as ``labels[j]``; `labels` is sorted.
    """
    unique_labels = np.union1d(y, predictions)
    n_labels = len(unique_labels)

    cm = np.zeros((n_labels, n_labels), dtype=int)
    for i, true_label in enumerate(unique_labels):
        masked = predictions[y == true_label]          # predictions for this true class
        for j, pred_label in enumerate(unique_labels):
            cm[i, j] = np.sum(masked == pred_label)

    return cm, unique_labels


def evaluation_report(cm: np.ndarray, labels: np.ndarray, digits: int = 3
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Print a per-class metrics table and return the metrics.

    The "1-vs-rest Acc" column is the accuracy of telling one class apart
    from all the others, ``(TP + TN) / total``. It is always at least the
    overall accuracy, so judge the model by precision, recall, F1 and the
    overall accuracy printed under the table.

    Args:
        cm: Confusion matrix from `confusion_matrix`.
        labels: Class labels for the rows / columns of `cm`.
        digits: Decimals shown in the table.

    Returns:
        ``(one_vs_rest_accuracies, recalls, precisions, f1_scores, macro_f1)``.
    """
    cm = np.asarray(cm)
    total = cm.sum()
    tp = np.diag(cm)                # correct predictions per class
    actual = cm.sum(axis=1)         # row sums: how many truly belong to each class
    predicted = cm.sum(axis=0)      # column sums: how many were predicted as each class

    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = np.where(predicted > 0, tp / predicted, 0.0)
        recalls = np.where(actual > 0, tp / actual, 0.0)
        f1_scores = np.where(precisions + recalls > 0,
                             2 * precisions * recalls / (precisions + recalls), 0.0)

    tn = total - actual - predicted + tp
    accuracies = (tp + tn) / total
    macro_f1 = float(f1_scores.mean())

    # ---------- table ----------
    f = lambda v: f"{v:.{digits}f}"
    headers = ["Class", "1-vs-rest Acc", "Precision", "Recall", "F1-score", "Support"]
    rows = [[str(lbl), f(accuracies[i]), f(precisions[i]), f(recalls[i]),
             f(f1_scores[i]), str(actual[i])] for i, lbl in enumerate(labels)]
    footer = ["Macro avg", f(accuracies.mean()), f(precisions.mean()),
              f(recalls.mean()), f(macro_f1), str(total)]

    widths = [max(len(r[k]) for r in [headers, *rows, footer]) for k in range(len(headers))]

    def border(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def line(cells):
        first = f" {cells[0]:<{widths[0]}} "
        rest = [f" {c:>{w}} " for c, w in zip(cells[1:], widths[1:])]
        return "│" + "│".join([first, *rest]) + "│"

    print(border("┌", "┬", "┐"))
    print(line(headers))
    print(border("├", "┼", "┤"))
    for r in rows:
        print(line(r))
    print(border("├", "┼", "┤"))
    print(line(footer))
    print(border("└", "┴", "┘"))
    print(f"Overall accuracy: {f(tp.sum() / total)}    Macro F1: {f(macro_f1)}")

    return accuracies, recalls, precisions, f1_scores, macro_f1


def show_confusion_matrix(cm: np.ndarray, labels: np.ndarray) -> None:
    """Plot a confusion matrix from `confusion_matrix` as a heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


def check_accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of predictions equal to the true labels."""
    n_test = len(labels)
    right = np.sum(prediction == labels)
    return right/n_test


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def stratified_group_k_fold(
        y: np.ndarray, k: int = 5, groups: np.ndarray | None = None, random_state: int | None = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split sample indices into `k` folds that keep groups whole and classes balanced.

    Every group (e.g. one original clip plus all its augmented copies) goes
    into exactly one fold, so no clip is ever validated against its own
    copy. Groups of each class are shuffled and then dealt out one by one to
    the fold that has the fewest samples of that class so far, which gives
    every fold about the same class proportions as the whole set.

    Args:
        y: Labels, shape ``(n_samples,)``.
        k: Number of folds, at least 2.
        groups: Group of each sample, e.g. ``Features.sources``. ``None``
            makes every sample its own group (plain stratified k-fold).
        random_state: Seed for shuffling the groups.

    Returns:
        ``k`` pairs ``(train_indices, test_indices)``.

    Raises:
        ValueError: If `k` is below 2 or above the number of groups, if
            `groups` differs in length from `y`, or if one group holds more
            than one class.
    """
    y = np.asarray(y)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError(f"'groups' has {len(groups)} entries but 'y' has {len(y)}")

    unique_groups, group_index = np.unique(groups, return_inverse=True)
    if not 2 <= k <= len(unique_groups):
        raise ValueError(f"'k'({k}) must be between 2 and the number of groups ({len(unique_groups)})")

    # One label per group
    group_label = np.empty(len(unique_groups), dtype=y.dtype)
    group_label[group_index] = y
    if np.any(group_label[group_index] != y):
        raise ValueError("Every group must contain a single class")
    group_size = np.bincount(group_index)

    # Deal each class's groups out to the folds
    rng = np.random.default_rng(random_state)
    group_fold = np.empty(len(unique_groups), dtype=int)
    fold_total = np.zeros(k, dtype=int)
    for label in np.unique(group_label):
        class_groups = rng.permutation(np.flatnonzero(group_label == label))
        fold_class = np.zeros(k, dtype=int)
        for g in class_groups:
            # fewest of this class; on a tie, the smallest fold overall
            fold = min(range(k), key=lambda f: (fold_class[f], fold_total[f]))
            group_fold[g] = fold
            fold_class[fold] += group_size[g]
            fold_total[fold] += group_size[g]

    sample_fold = group_fold[group_index]
    return [(np.flatnonzero(sample_fold != fold), np.flatnonzero(sample_fold == fold))
            for fold in range(k)]


def stratified_k_fold_cross_validation(
        model: Model, X: np.ndarray, y: np.ndarray, k: int = 5,
        groups: np.ndarray | None = None, random_state: int | None = 42
) -> Tuple[float, float]:
    """Estimate a model's accuracy with stratified (group) k-fold cross-validation.

    The model is trained on ``k - 1`` folds and scored on the remaining one,
    ``k`` times. Folds come from `stratified_group_k_fold`. Anything the
    model prints while fitting is silenced during the folds.

    Args:
        model: An untrained model; it is refitted on every fold.
        X: Features, shape ``(n_samples, n_features)``.
        y: Labels, shape ``(n_samples,)``.
        k: Number of folds, at least 2.
        groups: Group of each sample, e.g. ``Features.sources``. Pass it
            whenever `X` contains augmented copies.
        random_state: Seed for the fold assignment.

    Returns:
        ``(mean, std)`` of the fold accuracies.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    accuracies = []

    for train_indices, test_indices in stratified_group_k_fold(y, k, groups, random_state):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        with contextlib.redirect_stdout(io.StringIO()):
            trained = model.fit(X_train, y_train)
        prediction = trained.predict(X_test)
        accuracies.append(check_accuracy(prediction, y_test))

    accuracies = np.asarray(accuracies, dtype=float)
    return accuracies.mean(), accuracies.std()


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def create_grid_combinations(
        hyperparams: Dict[str, Sequence], excludes: List[Tuple[Tuple[str, str], Tuple]] | None = None
) -> List[dict]:
    """List every combination of hyperparameter values.

    Args:
        hyperparams: Values to try per parameter, e.g.
            ``{"kernel": ["linear", "rbf"], "C": [1, 10]}``.
        excludes: Rules that drop combinations, each
            ``((key_a, key_b), (value_a, value_b))``: when `key_a` equals
            `value_a`, only `value_b` is kept for `key_b`. For example
            ``(("kernel", "gamma"), ("linear", "scale"))`` tries the linear
            kernel once instead of once per gamma, because gamma does not
            affect it.

    Returns:
        One dict per combination.
    """
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    configs = []

    for combination in itertools.product(*values):
        cfg = dict(zip(keys, combination))
        skip = False
        if excludes is not None:
            for ex_keys, ex_values in excludes:
                if cfg[ex_keys[0]] == ex_values[0] and cfg[ex_keys[1]] != ex_values[1]:
                    skip = True
                    break

        if not skip:
            configs.append(cfg)

    return configs


def grid_search(
        hyperparams: Dict[str, Sequence], model: type, X: np.ndarray, y: np.ndarray,
        excludes: List | None = None, random_state: int = 42,
        groups: np.ndarray | None = None, k: int = 5,
) -> Tuple[dict, float]:
    """Find the hyperparameters with the best cross-validated accuracy.

    Args:
        hyperparams: Values to try per parameter (see `create_grid_combinations`).
        model: The model class, e.g. ``SVM`` or ``KNN`` (not an instance).
        X: Training features, shape ``(n_samples, n_features)``.
        y: Training labels.
        excludes: Combinations to skip (see `create_grid_combinations`).
        random_state: Seed for the folds, and for the model if its
            constructor accepts ``random_state``.
        groups: Group of each sample, e.g. ``Features.sources``. Pass it
            whenever `X` contains augmented copies.
        k: Number of cross-validation folds.

    Returns:
        ``(best_params, best_accuracy)``. On a tie the earlier combination wins.
    """
    combinations = create_grid_combinations(hyperparams, excludes)
    takes_seed = "random_state" in inspect.signature(model).parameters
    best_accuracy = 0.0
    best_params = {}

    for combination in tqdm(combinations, desc="searching for the best hyperparameter combination"):
        if takes_seed:
            my_model = model(random_state=random_state, **combination)
        else:
            my_model = model(**combination)
        mean, _ = stratified_k_fold_cross_validation(my_model, X, y, k, groups, random_state)
        if best_accuracy < mean:
            best_params = combination
            best_accuracy = mean

            print(f"new best params: {combination}")
            print(f"new best accuracy: {best_accuracy}")

    return best_params, best_accuracy


def search_best_duration(durations: List[float], model: Model, data: Dataset) -> Tuple[float, dict]:
    """Find the clip duration with the best cross-validated accuracy.

    For every duration, the clips are trimmed and clipped with
    `preprocessing.clip_all`, their features are extracted, and `model` is
    scored with `stratified_k_fold_cross_validation`. Each duration gets a
    new Dataset, so `data` itself is never changed.

    Args:
        durations: Longest clip lengths to try, in seconds, e.g.
            ``DURATIONS`` from config.py.
        model: An untrained model; it is refitted for every duration and fold.
        data: Clips to clip and score, normally the original training clips.
            The folds use no groups, so pass no augmented copies.

    Returns:
        ``(best_duration, history)``: the duration with the highest mean
        accuracy (on a tie the earlier one wins), and a dict from every
        duration to its mean accuracy.

    Note:
        The features go into the cross-validation unscaled. Scale them
        (see `preprocessing.z_score_standardization`) for results that
        match a model trained on scaled features.
    """
    history = {}
    best_accuracy = 0.0
    best_duration = None

    for duration in tqdm(durations):
        clipped = Dataset(clip_all(data.signals, max_duration=duration),
                          data.labels, data.names, data.sources, data.params)
        features = extract_all_features(clipped, desc=f"Extract Train Data with {duration}s Duration.")
        accuracy, _ = stratified_k_fold_cross_validation(model, features.X, features.labels)

        history[duration] = accuracy
        if accuracy > best_accuracy:
            best_duration = duration
            best_accuracy = accuracy

    print(f"The best duration is: {best_duration}, with accuracy: {best_accuracy}")
    return best_duration, history

# ---------------------------------------------------------------------------
# Comparing models across test sets and degradation levels
# ---------------------------------------------------------------------------

# Axis label per augmentation for `plot_robustness`.
_LEVEL_NAMES: Dict[str, str] = {"noise": "SNR (dB)", "mp3": "VBR quality", "downsample": "sample rate (Hz)"}


def test_sets(testing: dict) -> Iterator[Tuple[str, int | None, "Features"]]:
    """Yield every test set as ``(test_name, level, features)``.

    Args:
        testing: Dict from test name to its features, e.g.
            ``{"original": original["test"], "noise": noise["test"], ...}``.
            An augmentation maps to a dict with one `Features` per level
            (as `run_extraction` returns it); clean data maps to one `Features`.

    Yields:
        ``(test_name, level, features)``; `level` is ``None`` for clean data.
    """
    for test_name, data in testing.items():
        if isinstance(data, dict):                     # augmented: one Features per level
            for level, features in data.items():
                yield test_name, level, features
        else:
            yield test_name, None, data


def evaluate_models(models: Dict[str, Model], testing: dict, model_name: str
                    ) -> Tuple[pd.DataFrame, Dict[tuple, Tuple[np.ndarray, np.ndarray]]]:
    """Score every trained model on every test set and level.

    Args:
        models: Trained models keyed by training scenario, e.g.
            ``{"original": svm_a, "original+noise": svm_b}``.
        testing: Test sets, see `test_sets`.
        model_name: Name stored in the ``model`` column, e.g. ``"SVM"``.

    Returns:
        ``(results, matrices)``:

        - `results`: one row per (training scenario, test set, level), with
          columns ``model, train, test, level, accuracy, precision, recall,
          macro_f1``. Precision and recall are macro averages. Save it with
          ``results.to_csv(...)``.
        - `matrices`: ``(cm, labels)`` per ``(model_name, train, test, level)``,
          ready for `show_confusion_matrix`. Save it with ``pd.to_pickle(...)``.
    """
    rows, matrices = [], {}
    for train_scenario, model in tqdm(models.items(), f"Testing {model_name} models"):
        for test_name, level, test in test_sets(testing):
            cm, labels = confusion_matrix(model.predict(test.X), test.labels)
            with contextlib.redirect_stdout(io.StringIO()):      # no printed table per test
                _, recalls, precisions, _, macro_f1 = evaluation_report(cm, labels)
            matrices[(model_name, train_scenario, test_name, level)] = (cm, labels)
            rows.append({"model": model_name, "train": train_scenario, "test": test_name,
                         "level": level, "accuracy": np.trace(cm) / cm.sum(),
                         "precision": precisions.mean(), "recall": recalls.mean(),
                         "macro_f1": macro_f1})
    return pd.DataFrame(rows), matrices


def results_table(results: pd.DataFrame, model_name: str, metric: str = "macro_f1") -> pd.DataFrame:
    """Arrange one metric as training scenarios (rows) x test sets (columns).

    Columns are named like ``"original"``, ``"noise 30"``, ``"mp3 9"``, in
    the order they were tested (mild to severe per augmentation). Works on
    the output of `evaluate_models` or on the same table read back from CSV.

    Tip: in a notebook, colour it with
    ``.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format("{:.2f}")``.
    """
    df = results[results["model"] == model_name].copy()
    df["test set"] = [test if pd.isna(level) else f"{test} {level:g}"
                      for test, level in zip(df["test"], df["level"])]
    table = df.pivot(index="train", columns="test set", values=metric)
    return table.loc[df["train"].unique(), df["test set"].unique()]


def plot_robustness(results: pd.DataFrame, model_name: str, metric: str = "macro_f1") -> plt.Figure:
    """Plot a metric against degradation level, one line per training scenario.

    One panel per augmentation (noise, mp3, downsample); levels run from
    mild to severe in the order they were tested. Clean test results are
    left out (see `results_table`).

    Returns:
        The figure, e.g. to save with ``fig.savefig(...)``.
    """
    df = results[(results["model"] == model_name) & results["level"].notna()]
    tests = list(df["test"].unique())
    fig, axes = plt.subplots(1, len(tests), figsize=(5 * len(tests), 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, test_name in zip(axes, tests):
        part = df[df["test"] == test_name]
        levels = list(part["level"].unique())
        for train_scenario, rows in part.groupby("train", sort=False):
            ax.plot(range(len(levels)), rows.set_index("level").loc[levels, metric],
                    marker="o", label=train_scenario)
        ax.set_xticks(range(len(levels)), [f"{level:g}" for level in levels])
        ax.set_title(test_name)
        ax.set_xlabel(_LEVEL_NAMES.get(test_name, "level"))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric)
    axes[0].set_ylim(0, 1.05)
    # outside the panels, so it never covers a line
    axes[-1].legend(fontsize=8, title="trained on", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"{model_name}: {metric} per degradation level")
    fig.tight_layout()
    return fig


def show_table(model_name: str, metric: str, results: pd.DataFrame) -> None:
    """Display `results_table` coloured from red (0) to green (1), in a notebook.

    Args:
        model_name: Value of the ``model`` column to show, e.g. ``"SVM"``.
        metric: Column to show, e.g. ``"macro_f1"`` or ``"accuracy"``.
        results: Output of `evaluate_models`, or the same table read back
            from CSV.
    """
    styled = (results_table(results, model_name, metric)
              .style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None)
              .format("{:.2f}")
              .set_caption(f"{model_name} — {metric} (rows: trained on, columns: tested on)"))
    display(styled)

def show_confusion_grid(matrices: dict, model_name: str, test_name: str, level: int|str|None = None):
    """Plot one confusion matrix per training scenario, for one test set.

    The matrices are laid out four per row, each titled with the scenario
    the model was trained on.

    Args:
        matrices: Output of `evaluate_models`, keyed by
            ``(model_name, train, test, level)``.
        model_name: Model to show, e.g. ``"SVM"``.
        test_name: Test set to show, e.g. ``"original"`` or ``"noise"``.
        level: Degradation level of that test set, e.g. ``0`` for noise at
            0 dB. ``None`` for the clean test set.
    """
    keys = [k for k in matrices if k[0] == model_name and k[2] == test_name and k[3] == level]
    cols = 4
    rows = -(-len(keys) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        cm, labels = matrices[key]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"trained on: {key[1]}", fontsize=10)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"{model_name} confusion matrices — test: {test_name}" + ("" if level is None else f" {level}"))
    fig.tight_layout()
    plt.show()
