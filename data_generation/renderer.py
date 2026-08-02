"""三档渲染器：极简几何 / 中等纹理 / 类真实"""
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw


BALL_COLORS = [
    (220, 50, 50),    # 红
    (50, 180, 50),    # 绿
    (50, 100, 220),   # 蓝
    (220, 180, 50),   # 黄
]


def render_minimal(width, height, balls_state, bg_color=(240, 240, 240)):
    """极简几何：纯色圆 + 纯色背景"""
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    for i, ball in enumerate(balls_state):
        x, y, r = ball['x'], ball['y'], ball['radius']
        color = BALL_COLORS[i % len(BALL_COLORS)]
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=color,
            outline=(0, 0, 0),
            width=2
        )
        draw.text((x - 4, y - 6), str(ball['id']), fill=(255, 255, 255))

    return np.array(img)


def render_medium(width, height, balls_state, bg_color=(200, 200, 210)):
    """中等纹理：渐变 + 阴影"""
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # 背景渐变
    for y in range(height):
        ratio = y / height
        r = int(bg_color[0] * (1 - ratio * 0.3))
        g = int(bg_color[1] * (1 - ratio * 0.3))
        b = int(bg_color[2] * (1 - ratio * 0.3))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    for i, ball in enumerate(balls_state):
        x, y, r = ball['x'], ball['y'], ball['radius']
        base_color = BALL_COLORS[i % len(BALL_COLORS)]

        # 阴影
        shadow_offset = 4
        draw.ellipse(
            [x - r + shadow_offset, y - r + shadow_offset,
             x + r + shadow_offset, y + r + shadow_offset],
            fill=(100, 100, 100)
        )

        # 球体渐变
        for step in range(5):
            ratio = 1 - step * 0.15
            rr = r * ratio
            cr = min(255, int(base_color[0] * (1 + step * 0.1)))
            cg = min(255, int(base_color[1] * (1 + step * 0.1)))
            cb = min(255, int(base_color[2] * (1 + step * 0.1)))
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(cr, cg, cb))

        # 高光
        highlight_r = r * 0.3
        hx, hy = x - r * 0.3, y - r * 0.3
        draw.ellipse(
            [hx - highlight_r, hy - highlight_r, hx + highlight_r, hy + highlight_r],
            fill=(255, 255, 255)
        )

        draw.text((x - 4, y - 6), str(ball['id']), fill=(255, 255, 255))

    return np.array(img)


def render_realistic(width, height, balls_state, bg_color=(180, 190, 200)):
    """类真实：复杂背景、光照、运动模糊"""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 纹理背景
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    base = np.full_like(img, bg_color)
    img = cv2.addWeighted(base, 0.9, noise, 0.1, 0)

    # 背景物体
    for _ in range(random.randint(3, 8)):
        cx, cy = random.randint(0, width), random.randint(0, height)
        cr = random.randint(5, 30)
        color = tuple(random.randint(50, 200) for _ in range(3))
        cv2.circle(img, (cx, cy), cr, color, -1)

    # 线条纹理
    for _ in range(random.randint(2, 5)):
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        color = tuple(random.randint(100, 200) for _ in range(3))
        cv2.line(img, pt1, pt2, color, 1)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    for i, ball in enumerate(balls_state):
        x, y, r = int(ball['x']), int(ball['y']), int(ball['radius'])
        base_color = BALL_COLORS[i % len(BALL_COLORS)]

        # 运动模糊
        vx, vy = ball.get('vx', 0), ball.get('vy', 0)
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 50:
            blur_length = min(int(speed * 0.02), 10)
            angle = np.arctan2(vy, vx)
            for step in range(blur_length, 0, -2):
                bx = int(x - np.cos(angle) * step)
                by = int(y - np.sin(angle) * step)
                alpha = int(100 * (1 - step / blur_length))
                overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                od = ImageDraw.Draw(overlay)
                od.ellipse([bx - r, by - r, bx + r, by + r],
                           fill=(*base_color, alpha))
                pil_img = Image.alpha_composite(
                    pil_img.convert('RGBA'), overlay
                ).convert('RGB')
                draw = ImageDraw.Draw(pil_img)

        # 球体光照
        for step in range(8):
            ratio = 1 - step * 0.1
            rr = int(r * ratio)
            light = 1 + 0.15 * (1 - step / 8)
            cr = min(255, int(base_color[0] * light))
            cg = min(255, int(base_color[1] * light))
            cb = min(255, int(base_color[2] * light))
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(cr, cg, cb))

        draw.ellipse([x - r, y - r, x + r, y + r], outline=(30, 30, 30), width=2)

        # 高光
        hr = int(r * 0.25)
        hx, hy = x - int(r * 0.35), y - int(r * 0.35)
        draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(255, 255, 255))

        draw.text((x - 4, y - 6), str(ball['id']), fill=(255, 255, 255))

    # 全局光照
    img_np = np.array(pil_img).astype(np.float32)
    h, w = img_np.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    light_x, light_y = w * 0.3, h * 0.2
    dist = np.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
    light_map = 1.0 - dist / (np.sqrt(w**2 + h**2) * 0.8)
    light_map = np.clip(light_map, 0.7, 1.0)
    img_np = img_np * light_map[:, :, np.newaxis]

    return np.clip(img_np, 0, 255).astype(np.uint8)


RENDERERS = {
    'minimal': render_minimal,
    'medium': render_medium,
    'realistic': render_realistic,
}


def render_frame(width, height, balls_state, style='minimal'):
    """渲染单帧"""
    renderer = RENDERERS.get(style, render_minimal)
    return renderer(width, height, balls_state)


if __name__ == '__main__':
    balls = [
        {'x': 100, 'y': 100, 'radius': 25, 'id': 0, 'vx': 100, 'vy': 50},
        {'x': 300, 'y': 200, 'radius': 30, 'id': 1, 'vx': -80, 'vy': 120},
    ]

    for style in ['minimal', 'medium', 'realistic']:
        img = render_frame(448, 448, balls, style)
        cv2.imwrite(f'/tmp/test_{style}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"已生成 {style} 风格测试图")
