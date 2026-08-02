"""基线模型：恒定速度外推等"""
import numpy as np


class ConstantVelocityBaseline:
    """恒定速度外推基线：用上一帧速度预测下一帧"""

    def __init__(self):
        pass

    def predict(self, prev_velocities):
        """
        预测下一帧速度 = 上一帧速度

        Args:
            prev_velocities: (n_samples, 2) 上一帧速度 [vx, vy]

        Returns:
            (n_samples, 2) 预测的下一帧速度
        """
        return prev_velocities.copy()

    def evaluate(self, prev_velocities, true_next_velocities):
        """
        评估基线性能

        Args:
            prev_velocities: (n_samples, 2) 上一帧速度
            true_next_velocities: (n_samples, 2) 真实的下一帧速度

        Returns:
            dict: R², MSE 等指标
        """
        pred = self.predict(prev_velocities)

        # 计算 R²
        ss_res = np.sum((true_next_velocities - pred) ** 2)
        ss_tot = np.sum((true_next_velocities - true_next_velocities.mean(axis=0)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        mse = np.mean((true_next_velocities - pred) ** 2)

        return {
            'r2': float(r2),
            'mse': float(mse),
            'baseline_type': 'constant_velocity'
        }


class PositionExtrapolationBaseline:
    """位置外推基线：用当前位置和速度外推下一帧位置"""

    def __init__(self, dt=1/30.0):
        self.dt = dt

    def predict(self, positions, velocities):
        """
        预测下一帧位置 = 当前位置 + 速度 * dt

        Args:
            positions: (n_samples, 2) 当前位置
            velocities: (n_samples, 2) 当前速度

        Returns:
            (n_samples, 2) 预测的下一帧位置
        """
        return positions + velocities * self.dt

    def evaluate(self, positions, velocities, true_next_positions):
        """评估"""
        pred = self.predict(positions, velocities)

        ss_res = np.sum((true_next_positions - pred) ** 2)
        ss_tot = np.sum((true_next_positions - true_next_positions.mean(axis=0)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        mse = np.mean((true_next_positions - pred) ** 2)

        return {
            'r2': float(r2),
            'mse': float(mse),
            'baseline_type': 'position_extrapolation'
        }


class RandomBaseline:
    """随机基线"""

    def evaluate(self, y_true):
        """返回随机猜测的性能"""
        y_pred = np.random.randn(*y_true.shape) * y_true.std()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'r2': float(r2),
            'baseline_type': 'random'
        }


if __name__ == '__main__':
    # 测试
    np.random.seed(42)

    # 模拟数据：速度有持续性
    n = 100
    prev_vel = np.random.randn(n, 2)
    next_vel = prev_vel * 0.9 + np.random.randn(n, 2) * 0.1  # 强相关

    baseline = ConstantVelocityBaseline()
    result = baseline.evaluate(prev_vel, next_vel)
    print(f"恒定速度基线 R²: {result['r2']:.4f}")

    random_baseline = RandomBaseline()
    random_result = random_baseline.evaluate(next_vel)
    print(f"随机基线 R²: {random_result['r2']:.4f}")
