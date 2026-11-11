import numpy as np
import pandas as pd
from copy import copy, deepcopy
from pathlib import Path


class NeuralNetwork:

    def __init__(
        self,
        layer_sizes: list[int],
        use_softmax: bool = False,
        use_cel: bool = False,
        seed: int | None = None,
    ):
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
        self.hidden_sizes = layer_sizes[1:-1]
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(loc=0, scale=0.1, size=(layer_sizes[i + 1], layer_sizes[i] + 1)) for i in
                        range(len(layer_sizes) - 1)]
        self.use_softmax = use_softmax
        self.use_cel = use_cel

    # Prediction Methods
    @staticmethod
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x: np.ndarray[float]):
        exponentials = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exponentials / np.sum(exponentials, axis=0, keepdims=True)

    @staticmethod
    def mse(p: np.ndarray[float], y: np.ndarray[float]):
        return np.linalg.norm(p - y, 2) ** 2 / p.shape[0]

    @staticmethod
    def cel(p: np.ndarray[float], y: np.ndarray[float]):
        return -np.sum(y * np.log(p))

    def predict(self, x: np.ndarray[float]):
        current_vector = deepcopy(x)
        pre_activations = []
        post_activations = []
        for i, w in enumerate(self.weights):
            current_vector = np.concatenate((current_vector, np.ones(shape=(1, 1))))
            post_activations.append(deepcopy(current_vector))
            current_vector = w @ current_vector
            pre_activations.append(deepcopy(current_vector))
            if i != len(self.weights) - 1:
                current_vector = self.logistic(current_vector)
            else:
                if self.use_softmax:
                    current_vector = self.softmax(current_vector)
        return current_vector, pre_activations, post_activations

    # Backprop Methods
    @staticmethod
    def D_mse(p: np.ndarray[float], y: np.ndarray[float]):
        return 2 / p.shape[0] * (p - y).T

    @staticmethod
    def D_cel(p: np.ndarray[float], y: np.ndarray[float]):
        return - (y / p).T

    @staticmethod
    def D_identity(a: np.ndarray[float]):
        return np.identity(a.shape[0])

    @staticmethod
    def D_softmax(a: np.ndarray[float]):
        D = np.ndarray(shape=(a.shape[0], a.shape[0]), dtype=float)
        sm_vals = NeuralNetwork.softmax(a)
        for i, v1 in enumerate(a):
            for j, v2 in enumerate(a):
                if i == j:
                    delta = 1
                else:
                    delta = 0
                D[i, j] = sm_vals[i, 0] * (delta - sm_vals[j, 0])
        return D

    @staticmethod
    def D_logistic(a: np.ndarray[float]):
        return np.diag((NeuralNetwork.logistic(a) * (1 - NeuralNetwork.logistic(a))).T[0])

    @staticmethod
    def D_prev_output(w: np.ndarray[float]):
        return w

    @staticmethod
    def D_weights(w: np.ndarray[float], o: np.ndarray[float]):
        return np.kron(o.T, np.identity(w.shape[0]))

    def backpropagation(self, x: np.ndarray[float], y: np.ndarray[float], eta: float):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        prediction, pre_activations, post_activations = self.predict(x)
        output_count = prediction.shape[0]

        if self.use_cel:
            if not self.use_softmax:
                raise ValueError("Cross-entropy loss requires a softmax output")
            delta = prediction - y
        else:
            delta = 2.0 * (prediction - y) / output_count
            if self.use_softmax:
                delta = self.D_softmax(pre_activations[-1]) @ delta

        deltas = [None] * len(self.weights)
        deltas[-1] = delta
        for layer_index in range(len(self.weights) - 1, 0, -1):
            previous_delta = self.weights[layer_index][:, :-1].T @ deltas[layer_index]
            previous_activation = self.logistic(pre_activations[layer_index - 1])
            deltas[layer_index - 1] = previous_delta * previous_activation * (1.0 - previous_activation)

        gradients = [
            layer_delta @ post_activations[index].T
            for index, layer_delta in enumerate(deltas)
        ]
        for index, gradient in enumerate(gradients):
            self.weights[index] -= eta * gradient

        updated_prediction = self.predict(x)[0]
        if self.use_cel:
            return self.cel(updated_prediction, y)
        return self.mse(updated_prediction, y)


#I did use ai on the loaders and preprocessor but i just gave it the loaders from my old code and told it to put it in this. If you would prefer that I do it manually thats fine just ask, but the rest of the code is ai free
class DatasetLoaderPreprocessor:

    def __init__(self, data_dir: str = "."):
        script_dir = Path(__file__).resolve().parent
        requested_dir = Path(data_dir)
        self.data_dir = requested_dir if requested_dir.is_absolute() else script_dir / requested_dir
        self.class_names: dict[str, list[str]] = {}
        self.feature_median: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None

    def _read_data(self, filename: str, header: int | None = None) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return pd.read_csv(path, header=header, dtype=str)

    @staticmethod
    def _numeric_features(frame: pd.DataFrame) -> np.ndarray:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        return numeric.to_numpy(dtype=float)

    @staticmethod
    def _categorical_features(
        frame: pd.DataFrame,
        categories: dict[str, list[str]],
    ) -> np.ndarray:
        encoded_columns = []
        for column, values in categories.items():
            invalid_values = set(frame[column].dropna()) - set(values)
            if invalid_values:
                raise ValueError(f"Unexpected values in {column}: {sorted(invalid_values)}")
            categorical = pd.Series(pd.Categorical(frame[column], categories=values), name=column)
            encoded_columns.append(pd.get_dummies(categorical, dtype=float))
        return pd.concat(encoded_columns, axis=1).to_numpy(dtype=float)

    def _classification_target(self, labels: pd.Series, dataset_name: str) -> np.ndarray:
        labels = labels.astype(str)
        classes = sorted(labels.unique())
        self.class_names[dataset_name] = classes
        class_indices = {label: index for index, label in enumerate(classes)}
        indices = labels.map(class_indices).to_numpy(dtype=int)
        return np.eye(len(classes), dtype=float)[indices]

    @staticmethod
    def _regression_target(labels: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(labels, errors="coerce")
        if numeric.isna().any():
            raise ValueError("Regression target contains missing or non-numeric values")
        return numeric.to_numpy(dtype=float).reshape(-1, 1)

    def fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        """Fit standardization on training features and return their scaled values."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D feature array")
        self.feature_median = np.nanmedian(X, axis=0)
        self.feature_median = np.where(np.isnan(self.feature_median), 0.0, self.feature_median)
        X = np.where(np.isnan(X), self.feature_median, X)
        self.feature_mean = X.mean(axis=0)
        self.feature_scale = X.std(axis=0)
        self.feature_scale[self.feature_scale == 0.0] = 1.0
        return self.transform_features(X)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize features using statistics fitted on training data."""
        if self.feature_median is None or self.feature_mean is None or self.feature_scale is None:
            raise ValueError("Call fit_transform_features on training data first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.feature_mean.shape[0]:
            raise ValueError("X must be 2D and have the same feature count used during fitting")
        X = np.where(np.isnan(X), self.feature_median, X)
        return (X - self.feature_mean) / self.feature_scale

    def load_breast_cancer(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("breast-cancer-wisconsin.data")
        if data.shape[1] != 11:
            raise ValueError("Breast cancer data should have 10 features and a class column")
        X = self._numeric_features(data.iloc[:, 1:10])
        y = self._classification_target(data.iloc[:, 10], "breast_cancer")
        return X, y

    def load_car_evaluation(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("car.data")
        if data.shape[1] != 7:
            raise ValueError("Car evaluation data should have 6 features and a class column")
        features = data.iloc[:, :6].copy()
        features.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        X = self._categorical_features(features, {
            "buying": ["vhigh", "high", "med", "low"],
            "maint": ["vhigh", "high", "med", "low"],
            "doors": ["2", "3", "4", "5more"],
            "persons": ["2", "4", "more"],
            "lug_boot": ["small", "med", "big"],
            "safety": ["low", "med", "high"],
        })
        y = self._classification_target(data.iloc[:, 6], "car_evaluation")
        return X, y

    def load_congressional_votes(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("house-votes-84.data")
        if data.shape[1] != 17:
            raise ValueError("Congressional votes data should have a class and 16 vote columns")
        labels = data.iloc[:, 0]
        votes = data.iloc[:, 1:17].replace("?", "unknown")
        votes.columns = [f"vote_{i}" for i in range(votes.shape[1])]
        X = self._categorical_features(votes, {
            column: ["y", "n", "unknown"] for column in votes.columns
        })
        y = self._classification_target(labels, "congressional_votes")
        return X, y

    def load_abalone(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("abalone.data")
        if data.shape[1] != 9:
            raise ValueError("Abalone data should have 8 features and a rings target")
        sex = self._categorical_features(
            data.iloc[:, :1].set_axis(["sex"], axis=1),
            {"sex": ["M", "F", "I"]},
        )
        measurements = self._numeric_features(data.iloc[:, 1:8])
        X = np.concatenate((sex, measurements), axis=1)
        y = self._regression_target(data.iloc[:, 8])
        return X, y

    def load_computer_hardware(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("machine.data")
        if data.shape[1] != 10:
            raise ValueError("Computer hardware data should have 10 columns")
        # Vendor, model name, and ERP are excluded per machine.names; PRP is the target.
        X = self._numeric_features(data.iloc[:, 2:8])
        y = self._regression_target(data.iloc[:, 8])
        return X, y

    def load_forest_fires(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("forestfires.data", header=0)
        expected_columns = {
            "X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI",
            "temp", "RH", "wind", "rain", "area",
        }
        if set(data.columns) != expected_columns:
            raise ValueError("Forest fires data does not have the expected columns")
        categorical = self._categorical_features(data[["month", "day"]], {
            "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            "day": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        })
        numeric = self._numeric_features(data[["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]])
        X = np.concatenate((numeric, categorical), axis=1)
        y = self._regression_target(data["area"])
        return X, y


class Autoencoder:

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        learning_rate: float = 0.05,
        seed: int | None = None,
    ):
        if hidden_dim >= input_dim:
            raise ValueError("hidden_dim must be strictly smaller than input_dim")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.model = NeuralNetwork(
            [input_dim, hidden_dim, input_dim],
            use_softmax=False,
            use_cel=False,
            seed=seed,
        )

    def encode(self, x: np.ndarray[float]) -> np.ndarray[float]:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        h_in = np.concatenate((x, np.ones((1, 1))), axis=0)
        z = self.model.weights[0] @ h_in
        return self.model.logistic(z)

    def reconstruct(self, x: np.ndarray[float]) -> np.ndarray[float]:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        h = self.encode(x)
        h_with_bias = np.concatenate((h, np.ones((1, 1))), axis=0)
        return self.model.weights[1] @ h_with_bias

    def train(self, X: np.ndarray[float], epochs: int = 1000):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        for _ in range(epochs):
            for row in X:
                x = row.reshape(-1, 1)
                self.model.backpropagation(x, x, self.learning_rate)


def train_autoencoder(
    X: np.ndarray[float],
    hidden_dim: int,
    epochs: int = 1000,
    learning_rate: float = 0.05,
    seed: int | None = None,
):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    input_dim = X.shape[1]
    autoencoder = Autoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )
    autoencoder.train(X, epochs=epochs)
    return autoencoder

DATASET_LOADERS = {
    "breast_cancer": "load_breast_cancer",
    "car_evaluation": "load_car_evaluation",
    "congressional_votes": "load_congressional_votes",
    "abalone": "load_abalone",
    "computer_hardware": "load_computer_hardware",
    "forest_fires": "load_forest_fires",
}


def make_train_test_split(
    y: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 447,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)

    if y.ndim == 2 and y.shape[1] > 1:
        groups = [np.flatnonzero(np.argmax(y, axis=1) == label) for label in range(y.shape[1])]
    else:
        groups = [rng.permutation(y.shape[0])]

    train_parts = []
    test_parts = []
    for group in groups:
        group = rng.permutation(group)
        if y.ndim == 2 and y.shape[1] > 1:
            if group.size < 2:
                raise ValueError("at least two samples needed")
            test_count = min(group.size - 1, max(1, int(round(group.size * test_fraction))))
        else:
            if group.size < 2:
                raise ValueError("At least two samples needed")
            test_count = min(group.size - 1, max(1, int(round(group.size * test_fraction))))
        test_parts.append(group[:test_count])
        train_parts.append(group[test_count:])

    train_indices = rng.permutation(np.concatenate(train_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))
    return train_indices, test_indices


def _fit_network(
    model: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for index in rng.permutation(X.shape[0]):
            model.backpropagation(
                X[index].reshape(-1, 1),
                y[index].reshape(-1, 1),
                learning_rate,
            )


def _predict_network(model: NeuralNetwork, X: np.ndarray) -> np.ndarray:
    return np.vstack([
        model.predict(row.reshape(-1, 1))[0].reshape(1, -1)
        for row in X
    ])


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_classification: bool,
) -> dict[str, float]:
    if is_classification:
        probabilities = np.clip(y_pred, 1e-12, 1.0)
        return {
            "accuracy": float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))),
            "cross_entropy": float(-np.mean(np.sum(y_true * np.log(probabilities), axis=1))),
        }

    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    total_sum_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = 1.0 - float(np.sum(errors ** 2)) / total_sum_squares if total_sum_squares else 0.0
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "r_squared": r_squared,
    }


#Use this for the 5x2
#You'll get it

def run_experiment_split(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.01,
    hidden_sizes: tuple[int, int] | None = None,
    encoder_dim: int | None = None,
    seed: int = 447,
) -> list[dict[str, str | int | float]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError("Both training and test splits must contain samples")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("no overlap allowed")
    if (
        np.any(train_indices < 0)
        or np.any(test_indices < 0)
        or np.any(train_indices >= X.shape[0])
        or np.any(test_indices >= X.shape[0])
    ):
        raise IndexError("out of bounds")
    X_train_raw, X_test_raw = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    is_classification = y.ndim == 2 and y.shape[1] > 1

    preprocessor = DatasetLoaderPreprocessor()
    X_train = preprocessor.fit_transform_features(X_train_raw)
    X_test = preprocessor.transform_features(X_test_raw)

    if is_classification:
        train_targets = y_train
        is_softmax = True
        is_cross_entropy = True
        target_mean = np.zeros((1, y.shape[1]))
        target_scale = np.ones((1, y.shape[1]))
    else:
        target_mean = y_train.mean(axis=0, keepdims=True)
        target_scale = y_train.std(axis=0, keepdims=True)
        target_scale[target_scale == 0.0] = 1.0
        train_targets = (y_train - target_mean) / target_scale
        is_softmax = False
        is_cross_entropy = False

    input_dim = X_train.shape[1]
    output_dim = y.shape[1]
    if hidden_sizes is None:
        hidden1 = max(4, min(32, input_dim))
        hidden2 = max(2, hidden1 // 2)
    else:
        hidden1, hidden2 = hidden_sizes
        if hidden1 < 1 or hidden2 < 1:
            raise ValueError("must be positive")
    if encoder_dim is None:
        encoder_dim = max(1, min(input_dim - 1, input_dim // 2))
    if not 0 < encoder_dim < input_dim:
        raise ValueError("out of bounds")

    model_specs: list[tuple[str, NeuralNetwork]] = [
        (
            "linear_baseline",
            NeuralNetwork(
                [input_dim, output_dim],
                use_softmax=is_softmax,
                use_cel=is_cross_entropy,
                seed=seed,
            ),
        ),
        (
            "two_hidden_layer",
            NeuralNetwork(
                [input_dim, hidden1, hidden2, output_dim],
                use_softmax=is_softmax,
                use_cel=is_cross_entropy,
                seed=seed + 1,
            ),
        ),
    ]

    autoencoder = train_autoencoder(
        X_train,
        hidden_dim=encoder_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed + 2,
    )
    pretrained_model = NeuralNetwork(
        [input_dim, encoder_dim, hidden2, output_dim],
        use_softmax=is_softmax,
        use_cel=is_cross_entropy,
        seed=seed + 3,
    )
    pretrained_model.weights[0] = autoencoder.model.weights[0].copy()
    model_specs.append(("autoencoder_pretrained", pretrained_model))

    results: list[dict[str, str | int | float]] = []
    for model_index, (model_name, model) in enumerate(model_specs):
        _fit_network(
            model,
            X_train,
            train_targets,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed + 10 + model_index,
        )

        train_predictions = _predict_network(model, X_train)
        test_predictions = _predict_network(model, X_test)
        if not is_classification:
            train_predictions = train_predictions * target_scale + target_mean
            test_predictions = test_predictions * target_scale + target_mean

        row: dict[str, str | int | float] = {
            "dataset": dataset_name,
            "model": model_name,
            "train_size": int(len(train_indices)),
            "test_size": int(len(test_indices)),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden1": hidden1 if model_name == "two_hidden_layer" else (
                encoder_dim if model_name == "autoencoder_pretrained" else 0
            ),
            "hidden2": hidden2 if model_name != "linear_baseline" else 0,
        }
        row.update({
            f"train_{metric}": value
            for metric, value in _calculate_metrics(y_train, train_predictions, is_classification).items()
        })
        row.update({
            f"test_{metric}": value
            for metric, value in _calculate_metrics(y_test, test_predictions, is_classification).items()
        })
        results.append(row)

    return results

#You'll need to edit this to for the 5x2 and tests
def experiment_runner(
    data_dir: str = ".",
    output_file: str = "experiment_results.csv",
    test_fraction: float = 0.2,
    epochs: int = 100,
    learning_rate: float = 0.01,
    seed: int = 447,
) -> list[dict[str, str | int | float]]:
    loader = DatasetLoaderPreprocessor(data_dir=data_dir)
    all_results: list[dict[str, str | int | float]] = []

    for dataset_index, (dataset_name, loader_method) in enumerate(DATASET_LOADERS.items()):
        X, y = getattr(loader, loader_method)()
        train_indices, test_indices = make_train_test_split(
            y,
            test_fraction=test_fraction,
            seed=seed + dataset_index,
        )
        dataset_results = run_experiment_split(
            dataset_name,
            X,
            y,
            train_indices,
            test_indices,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed + dataset_index * 100,
        )
        all_results.extend(dataset_results)
        for result in dataset_results:
            metric_text = ", ".join(
                f"{key}={value:.4f}"
                for key, value in result.items()
                if key.startswith("test_") and key != "test_size"
            )
            print(f"{dataset_name} | {result['model']} | {metric_text}")

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    return all_results


if __name__ == '__main__':
    experiment_runner()
