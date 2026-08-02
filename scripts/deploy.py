"""部署脚本：上传代码到服务器并运行测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from server import ServerManager


def deploy_and_test():
    server = ServerManager()
    server.connect()

    print("\n" + "=" * 60)
    print("部署代码到服务器")
    print("=" * 60)

    # 创建远程目录
    server.run(f'mkdir -p {server.work_dir}')

    # 上传项目文件
    local_project = Path(__file__).parent.parent
    print(f"\n上传项目文件...")

    # 需要上传的目录
    dirs_to_upload = ['data_generation', 'models', 'scripts', 'configs']
    for dir_name in dirs_to_upload:
        local_dir = local_project / dir_name
        if local_dir.exists():
            remote_dir = f"{server.work_dir}/{dir_name}"
            print(f"  上传 {dir_name}/...")
            server.upload_dir(local_dir, remote_dir)

    # 上传 requirements.txt
    req_file = local_project / 'requirements.txt'
    if req_file.exists():
        server.upload(str(req_file), f"{server.work_dir}/requirements.txt")
        print("  上传 requirements.txt")

    print("\n部署完成")

    # 运行测试
    print("\n" + "=" * 60)
    print("在服务器上运行 Day 1 测试")
    print("=" * 60)

    # 先测试数据生成（不需要模型）
    print("\n--- 测试数据生成 ---")
    out, err, code = server.run(
        f'cd {server.work_dir} && {server.python} -c "'
        'from data_generation.physics_sim import generate_scene; '
        'from data_generation.renderer import render_frame; '
        'import numpy as np; '
        'data = generate_scene(seed=42); '
        'print(f\\"物理模拟: {data[\\"num_balls\\"]} 个球\\"); '
        'balls = [{**data[\\"balls\\"][0][\\"trajectory\\"][0], \\"id\\": 0, \\"radius\\": data[\\"balls\\"][0][\\"radius\\"]}]; '
        'img = render_frame(448, 448, balls, \\"minimal\\"); '
        'print(f\\"渲染: shape {img.shape}\\")'
        '"'
    )
    print(out)
    if err:
        print("ERR:", err)

    server.close()


if __name__ == '__main__':
    deploy_and_test()
