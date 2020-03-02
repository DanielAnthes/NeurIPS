import time
import os
import subprocess
from Neurosmash import Environment as ENV

UNITY_PATH = r".\Windows\NeuroSmashLite.exe"

base_port = 10000
ip = "127.0.0.1"
timescale = "5"
resolution = "64"

envs = []
for i in range(4):
    port = base_port + i
    print(f""" Opening Environment
IP        : {ip}
Port      : {port}
TimeScale : {timescale}
Resolution: {resolution}
    """)
    subprocess.Popen([
        UNITY_PATH,
        "-batchmode",
        "-I", # --ip
        ip,
        "-P", # --port
        str(port),
        "-T", # --timescale
        timescale,
        "-R", # --resolution
        resolution
    ])
    time.sleep(1)
    envs.append(ENV(ip=ip, port=port, timescale=timescale, size=resolution))

print("Opened Environments. Sleeping...")
time.sleep(10)
print("Closing Environments.")
for e in envs:
    e.quit()
