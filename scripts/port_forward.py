import subprocess
import os
import sys
import signal
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_DIR = os.path.join(BASE_DIR, ".pids")

os.makedirs(PID_DIR, exist_ok=True)

PORT_FORWARDS = [
    # name, namespace, service, local_port, remote_port
    ("api",        "heart-mlops", "heart-api-lb", 8000, 80),
    ("mlflow",     "heart-mlops", "mlflow",       5000, 5000),
    ("minio-s3",   "heart-mlops", "minio",        9000, 9000),
    ("minio-ui",   "heart-mlops", "minio",        9001, 9001),
    ("prometheus", "monitoring",  "prometheus",   9090, 9090),
    ("grafana",    "monitoring",  "grafana",      3000, 3000),
]

def pid_file(name):
    return os.path.join(PID_DIR, f"{name}.pid")

def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except:
        return False

def start_forward(name, namespace, service, local_port, remote_port):
    pid_path = pid_file(name)

    if os.path.exists(pid_path):
        with open(pid_path) as f:
            old_pid = int(f.read().strip())
        if is_running(old_pid):
            print(f"[SKIP] {name} already running (PID {old_pid})")
            return
        else:
            os.remove(pid_path)

    cmd = [
        "kubectl", "port-forward",
        "-n", namespace,
        f"svc/{service}",
        f"{local_port}:{remote_port}",
        "--address", "127.0.0.1"
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )

    with open(pid_path, "w") as f:
        f.write(str(proc.pid))

    print(f"[STARTED] {name:<12} -> http://127.0.0.1:{local_port} (PID {proc.pid})")

def stop_all():
    print("Stopping all port-forwards...")
    for file in os.listdir(PID_DIR):
        if not file.endswith(".pid"):
            continue
        path = os.path.join(PID_DIR, file)
        name = file.replace(".pid", "")
        with open(path) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[STOPPED] {name} (PID {pid})")
        except:
            print(f"[FAILED] {name} (PID {pid})")
        os.remove(path)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_all()
        return

    print("Starting Kubernetes port-forwards...\n")

    for pf in PORT_FORWARDS:
        start_forward(*pf)

    print("\nCentral endpoints:")
    print(" API        : http://127.0.0.1:8000")
    print(" MLflow     : http://127.0.0.1:5000")
    print(" MinIO S3   : http://127.0.0.1:9000")
    print(" MinIO UI   : http://127.0.0.1:9001")
    print(" Prometheus : http://127.0.0.1:9090")
    print(" Grafana    : http://127.0.0.1:3000")
    print("\nPress Ctrl+C to exit (port-forwards stay running)")

if __name__ == "__main__":
    main()
