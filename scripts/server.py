"""服务器管理工具：SSH 连接、文件传输、远程执行"""
import os
import paramiko
import stat
from pathlib import Path


class ServerManager:
    def __init__(self):
        # 凭据通过环境变量提供，切勿硬编码真实密码
        self.host = os.environ.get('AUTODL_HOST', '')
        self.port = int(os.environ.get('AUTODL_PORT', '22'))
        self.username = os.environ.get('AUTODL_USERNAME', 'root')
        self.password = os.environ.get('AUTODL_PASSWORD', '')
        self.python = '/root/miniconda3/bin/python'
        self.work_dir = '/root/autodl-tmp/vlm_kinematics_probing'
        self.client = None
        self.sftp = None

    def connect(self):
        """建立 SSH 连接"""
        if not self.host or not self.password:
            raise RuntimeError(
                "缺少服务器凭据，请设置环境变量 AUTODL_HOST 和 AUTODL_PASSWORD"
            )
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=30
        )
        self.sftp = self.client.open_sftp()
        print(f"已连接: {self.host}")

    def close(self):
        """关闭连接"""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()

    def run(self, cmd, timeout=300):
        """执行远程命令，返回 (stdout, stderr, exit_code)"""
        stdin, stdout, stderr = self.client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        code = stdout.channel.recv_exit_status()
        return out, err, code

    def run_python(self, code_str, timeout=300):
        """在远程执行 Python 代码"""
        # 写入临时文件再执行，避免引号问题
        remote_tmp = '/tmp/_remote_exec.py'
        self.write_file(remote_tmp, code_str)
        out, err, code = self.run(f'{self.python} {remote_tmp}', timeout=timeout)
        return out, err, code

    def write_file(self, remote_path, content):
        """写入远程文件"""
        with self.sftp.file(remote_path, 'w') as f:
            f.write(content)

    def read_file(self, remote_path):
        """读取远程文件"""
        with self.sftp.file(remote_path, 'r') as f:
            return f.read().decode('utf-8')

    def upload(self, local_path, remote_path):
        """上传文件"""
        self.sftp.put(local_path, remote_path)

    def upload_dir(self, local_dir, remote_dir):
        """递归上传目录"""
        local_dir = Path(local_dir)
        for item in local_dir.rglob('*'):
            if item.is_file():
                rel = item.relative_to(local_dir)
                remote_path = f"{remote_dir}/{rel.as_posix()}"
                self._mkdir_p(str(Path(remote_path).parent).replace('\\', '/'))
                self.sftp.put(str(item), remote_path)
                print(f"  上传: {rel}")

    def download(self, remote_path, local_path):
        """下载文件"""
        self.sftp.get(remote_path, local_path)

    def _mkdir_p(self, remote_dir):
        """递归创建远程目录"""
        dirs = []
        while remote_dir and remote_dir != '/':
            dirs.append(remote_dir)
            remote_dir = str(Path(remote_dir).parent).replace('\\', '/')
        for d in reversed(dirs):
            try:
                self.sftp.mkdir(d)
            except IOError:
                pass

    def exists(self, remote_path):
        """检查远程路径是否存在"""
        try:
            self.sftp.stat(remote_path)
            return True
        except IOError:
            return False


def main():
    """环境验证入口"""
    server = ServerManager()
    server.connect()

    print("\n" + "=" * 60)
    print("服务器环境验证")
    print("=" * 60)

    # 1. PyTorch 检查
    print("\n【PyTorch 环境】")
    out, err, _ = server.run_python('''
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
''')
    print(out)
    if err:
        print("ERR:", err)

    # 2. 依赖检查
    print("\n【Python 依赖】")
    out, err, _ = server.run_python('''
deps = ["transformers", "accelerate", "pymunk", "pygame", "cv2", "PIL",
        "sklearn", "numpy", "tqdm", "yaml", "matplotlib", "qwen_vl_utils"]
missing = []
for d in deps:
    try:
        m = __import__(d)
        v = getattr(m, "__version__", "ok")
        print(f"  OK: {d} ({v})")
    except ImportError:
        print(f"  MISSING: {d}")
        missing.append(d)
if missing:
    print(f"\\n需要安装: {' '.join(missing)}")
else:
    print("\\n所有依赖已就绪")
''')
    print(out)

    # 3. 磁盘空间
    print("\n【磁盘空间】")
    out, _, _ = server.run('df -h /root/autodl-tmp /')
    print(out)

    # 4. 显存状态
    print("\n【GPU 状态】")
    out, _, _ = server.run('nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv')
    print(out)

    # 5. 检查 Ollama
    print("\n【Ollama 进程】")
    out, _, _ = server.run('ps aux | grep ollama | grep -v grep || echo "Ollama 未运行"')
    print(out if out.strip() else "Ollama 未运行")

    server.close()
    print("\n环境验证完成")


if __name__ == '__main__':
    main()
