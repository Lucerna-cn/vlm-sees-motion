import os
import paramiko
import sys

def check_server():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        host = os.environ.get('AUTODL_HOST', '')
        port = int(os.environ.get('AUTODL_PORT', '22'))
        username = os.environ.get('AUTODL_USERNAME', 'root')
        password = os.environ.get('AUTODL_PASSWORD', '')
        if not host or not password:
            print("缺少服务器凭据，请设置环境变量 AUTODL_HOST 和 AUTODL_PASSWORD")
            return False

        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            timeout=15
        )
        print("=" * 60)
        print("SSH 连接成功")
        print("=" * 60)

        # 检查 GPU
        stdin, stdout, stderr = client.exec_command('nvidia-smi')
        gpu_info = stdout.read().decode()
        print("\n【GPU 信息】")
        print(gpu_info)

        # 检查 Python 环境
        stdin, stdout, stderr = client.exec_command('python --version && which python')
        python_info = stdout.read().decode()
        print("【Python 环境】")
        print(python_info)

        # 检查 PyTorch
        stdin, stdout, stderr = client.exec_command('python -c "import torch; print(f\'PyTorch: {torch.__version__}\'); print(f\'CUDA available: {torch.cuda.is_available()}\'); print(f\'CUDA version: {torch.version.cuda}\')"')
        torch_info = stdout.read().decode()
        torch_err = stderr.read().decode()
        print("【PyTorch】")
        print(torch_info)
        if torch_err:
            print("Error:", torch_err)

        # 检查 transformers
        stdin, stdout, stderr = client.exec_command('python -c "import transformers; print(f\'Transformers: {transformers.__version__}\')"')
        tf_info = stdout.read().decode()
        print("【Transformers】")
        print(tf_info)

        # 检查磁盘空间
        stdin, stdout, stderr = client.exec_command('df -h /root')
        disk_info = stdout.read().decode()
        print("【磁盘空间】")
        print(disk_info)

        # 检查关键依赖
        stdin, stdout, stderr = client.exec_command('python -c "import pymunk, sklearn, cv2, PIL, numpy, tqdm, yaml; print(\'All dependencies OK\')" 2>&1')
        deps_info = stdout.read().decode()
        print("【依赖检查】")
        print(deps_info)

        client.close()
        return True

    except Exception as e:
        print(f"连接失败: {e}")
        return False

if __name__ == '__main__':
    check_server()
