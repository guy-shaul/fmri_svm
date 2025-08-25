import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shura

# Setup logger
log = shura.get_logger(name= "Model", level="DEBUG", to_file= False, filename="model.log", file_format="log")



class SVMClassifier:
    def __init__(self, kernel='sigmoid', scale=True, k_folds=10, random_state=42):
        self.kernel = kernel
        self.scale = scale
        self.k_folds = k_folds
        self.random_state = random_state
        self.scaler = StandardScaler() if scale else None
        self.model = None

    def prepare_input(self, X, y):
        try:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1)

            if self.scale:
                X = self.scaler.fit_transform(X)

            return X, y
        except Exception as e:
            log.error(f"Error in prepare_input: {e}")
            raise

    def train_and_evaluate(self, X, y):
        log.info(f"======= Training SVM classifier =======")
        log.debug(
            f"Model params: kernel={self.kernel}, scale={self.scale}, k_folds={self.k_folds}, random_state={self.random_state}")

        X, y = self.prepare_input(X, y)
        try:
            kf = KFold(n_splits=self.k_folds, shuffle=False)

            fold_accuracies = []
            fold_reports = []
            best_accuracy = 0
            best_conf_matrix = None
            sum_conf_matrix = None

            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                log.info(f"Training fold {fold}/{self.k_folds}")

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                try:
                    clf = SVC(kernel=self.kernel, decision_function_shape='ovr', random_state=self.random_state)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred)
                    matrix = confusion_matrix(y_test, y_pred)

                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_conf_matrix = matrix

                    sum_conf_matrix = matrix if sum_conf_matrix is None else sum_conf_matrix + matrix

                    fold_accuracies.append(acc)
                    fold_reports.append(report)

                    log.info(f"Fold {fold} Accuracy: {acc:.4f}")
                except Exception as e:
                    log.warning(f"Error in fold {fold}: {e}")
                    continue

            log.info(f"Cross-validation results: Mean Accuracy: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f} | Best Fold Accuracy: {best_accuracy:.3f}")

            return fold_accuracies, fold_reports, best_conf_matrix, sum_conf_matrix

        except Exception as e:
            log.error(f"Unexpected error during training: {e}")
            return None, None, None, None
        finally:
            print("-------------------------------")