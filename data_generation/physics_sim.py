"""2D 物理模拟：基于 pymunk 的小球运动仿真"""
import pymunk
import numpy as np
import random


class BallScene:
    """单场景物理模拟器：1-3 个球在边界内运动"""

    def __init__(self, width=448, height=448, num_balls=None, gravity=None, seed=None):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_balls = num_balls if num_balls else random.randint(1, 3)

        # 随机重力
        if gravity is None:
            gravity = random.choice([(0, 900), (0, -900), (0, 0)])
        self.gravity = gravity

        # 物理空间
        self.space = pymunk.Space()
        self.space.gravity = gravity

        # 边界
        self._add_boundaries()

        # 创建小球
        self.balls = []
        self._create_balls()

        # 记录轨迹
        self.trajectories = [[] for _ in range(self.num_balls)]
        self.collision_events = []
        self._current_frame = 0

        # 碰撞检测
        self._setup_collision_handler()

    def _add_boundaries(self):
        """添加四面墙壁"""
        static_body = self.space.static_body
        walls = [
            pymunk.Segment(static_body, (0, 0), (self.width, 0), 2),
            pymunk.Segment(static_body, (0, self.height), (self.width, self.height), 2),
            pymunk.Segment(static_body, (0, 0), (0, self.height), 2),
            pymunk.Segment(static_body, (self.width, 0), (self.width, self.height), 2),
        ]
        for wall in walls:
            wall.elasticity = random.uniform(0.7, 0.95)
            wall.friction = random.uniform(0.1, 0.5)
        self.space.add(*walls)

    def _create_balls(self):
        """创建随机参数的小球"""
        for i in range(self.num_balls):
            radius = random.uniform(15, 35)
            mass = radius * 0.1
            moment = pymunk.moment_for_circle(mass, 0, radius)

            body = pymunk.Body(mass, moment)
            body.position = (
                random.uniform(radius + 20, self.width - radius - 20),
                random.uniform(radius + 20, self.height - radius - 20)
            )
            body.velocity = (
                random.uniform(-300, 300),
                random.uniform(-300, 300)
            )

            shape = pymunk.Circle(body, radius)
            shape.elasticity = random.uniform(0.5, 0.95)
            shape.friction = random.uniform(0.1, 0.5)
            shape.collision_type = i + 1

            self.space.add(body, shape)
            self.balls.append({
                'body': body,
                'shape': shape,
                'radius': radius,
                'id': i
            })

    def _setup_collision_handler(self):
        """设置碰撞事件记录（pymunk 7.x 兼容）"""
        # pymunk 7.x 使用不同的碰撞处理 API
        # 简化处理：在 step 中检测碰撞
        pass

    def _detect_collisions(self):
        """在 step 中检测球-球碰撞"""
        for i in range(self.num_balls):
            for j in range(i + 1, self.num_balls):
                pos_i = self.balls[i]['body'].position
                pos_j = self.balls[j]['body'].position
                dist = ((pos_i.x - pos_j.x)**2 + (pos_i.y - pos_j.y)**2)**0.5
                min_dist = self.balls[i]['radius'] + self.balls[j]['radius']

                # 检测是否发生碰撞（距离小于半径和，且之前帧距离更大）
                if dist < min_dist * 1.05:  # 5% 容差
                    # 检查是否为新碰撞
                    is_new = True
                    for evt in self.collision_events[-5:]:  # 检查最近5个事件
                        if set(evt['shapes']) == {i + 1, j + 1}:
                            is_new = False
                            break

                    if is_new:
                        self.collision_events.append({
                            'frame': self._current_frame,
                            'shapes': [i + 1, j + 1]
                        })

    def step(self, dt=1/30.0):
        """推进一步物理模拟并记录状态"""
        self.space.step(dt)

        # 检测碰撞
        self._detect_collisions()

        for i, ball in enumerate(self.balls):
            body = ball['body']
            pos = body.position
            vel = body.velocity
            acc = (0, 0)
            if len(self.trajectories[i]) > 0:
                prev_vel = (self.trajectories[i][-1]['vx'], self.trajectories[i][-1]['vy'])
                acc = ((vel.x - prev_vel[0]) / dt, (vel.y - prev_vel[1]) / dt)

            self.trajectories[i].append({
                'x': float(pos.x),
                'y': float(pos.y),
                'vx': float(vel.x),
                'vy': float(vel.y),
                'ax': float(acc[0]),
                'ay': float(acc[1]),
                'radius': float(ball['radius'])
            })

        self._current_frame += 1

    def simulate(self, num_frames=8, dt=1/30.0):
        """模拟完整序列"""
        for _ in range(num_frames):
            self.step(dt)
        return self.trajectories

    def get_trajectory_data(self):
        """获取结构化轨迹数据"""
        return {
            'num_balls': self.num_balls,
            'gravity': list(self.gravity),
            'width': self.width,
            'height': self.height,
            'balls': [
                {
                    'id': i,
                    'radius': self.balls[i]['radius'],
                    'trajectory': self.trajectories[i]
                }
                for i in range(self.num_balls)
            ],
            'collisions': self.collision_events
        }


def generate_scene(width=448, height=448, num_frames=8, seed=None, **kwargs):
    """生成单个场景，返回轨迹数据"""
    scene = BallScene(width, height, seed=seed, **kwargs)
    scene.simulate(num_frames)
    return scene.get_trajectory_data()


if __name__ == '__main__':
    data = generate_scene(seed=42)
    print(f"生成场景: {data['num_balls']} 个球")
    for ball in data['balls']:
        traj = ball['trajectory']
        print(f"  球 {ball['id']}: 半径 {ball['radius']:.1f}, "
              f"起始 ({traj[0]['x']:.1f}, {traj[0]['y']:.1f}), "
              f"速度 ({traj[0]['vx']:.1f}, {traj[0]['vy']:.1f})")
