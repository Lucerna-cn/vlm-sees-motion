"""线性/非线性探针实现"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


class RidgeProbe:
    """Ridge 回归探针"""

    def __init__(self, alphas=None):
        if alphas is None:
            alphas = [0.1, 1.0, 10.0, 100.0]
        self.alphas = alphas
        self.model = None

    def fit(self, X, y):
        """训练探针"""
        self.model = RidgeCV(alphas=self.alphas)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def fit_evaluate(self, X, y, test_size=0.2, random_state=42):
        """训练并评估"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # 计算相对误差
        relative_error = np.sqrt(mse) / (np.std(y_test) + 1e-8)

        return {
            'r2': float(r2),
            'mse': float(mse),
            'relative_error': float(relative_error),
            'alpha': float(self.model.alpha_),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }


class MLPProbe:
    """2 层 MLP 探针"""

    def __init__(self, hidden_dim=256, max_iter=500):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y):
        """训练探针"""
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim),
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def fit_evaluate(self, X, y, test_size=0.2, random_state=42):
        """训练并评估"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        relative_error = np.sqrt(mse) / (np.std(y_test) + 1e-8)

        return {
            'r2': float(r2),
            'mse': float(mse),
            'relative_error': float(relative_error),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }


class ControlTaskProbe:
    """选择性测试探针：预测随机打乱标签"""

    def __init__(self, base_probe=None):
        if base_probe is None:
            base_probe = RidgeProbe()
        self.base_probe = base_probe

    def run_control_task(self, X, y, n_shuffles=5):
        """
        运行 control task

        Returns:
            dict: 包含真实标签和打乱标签的 R² 对比
        """
        # 真实标签
        real_result = self.base_probe.fit_evaluate(X, y)

        # 打乱标签
        shuffled_r2s = []
        for i in range(n_shuffles):
            y_shuffled = np.random.permutation(y)
            shuffle_result = self.base_probe.fit_evaluate(X, y_shuffled)
            shuffled_r2s.append(shuffle_result['r2'])

        return {
            'real_r2': real_result['r2'],
            'shuffled_r2_mean': float(np.mean(shuffled_r2s)),
            'shuffled_r2_std': float(np.std(shuffled_r2s)),
            'selectivity': real_result['r2'] - np.mean(shuffled_r2s)
        }


if __name__ == '__main__':
    # 测试
    np.random.seed(42)
    X = np.random.randn(100, 64)
    y = X[:, 0] * 2 + np.random.randn(100) * 0.1

    probe = RidgeProbe()
    result = probe.fit_evaluate(X, y)
    print(f"Ridge R²: {result['r2']:.4f}")

    control = ControlTaskProbe()
    control_result = control.run_control_task(X, y)
    print(f"Control task - Real R²: {control_result['real_r2']:.4f}, "
          f"Shuffled R²: {control_result['shuffled_r2_mean']:.4f}")
