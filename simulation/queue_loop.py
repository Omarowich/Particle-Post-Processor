"""Keep running the batch queue with FRESH code for every round.

    python queue_loop.py            (4 runs at the same time)
    python queue_loop.py 5

Each round starts sim_runner.py --queue in a new Python process, so code changes made
while this is running (new potentials etc.) are used from the next round on.
Stop: Ctrl+C, or create a file named STOP_LOOP in the queue folder.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = r"D:\Work\AIRPODS Paper 2627\Data\_batches\queue"
W = sys.argv[1] if len(sys.argv) > 1 else "4"

while not os.path.exists(os.path.join(QUEUE, "STOP_LOOP")):
    todo = [f for f in os.listdir(QUEUE) if f.lower().endswith((".csv", ".json"))]
    if todo:
        print(time.strftime("%H:%M"), f"round: {len(todo)} file(s) in the queue")
        subprocess.run([sys.executable, os.path.join(HERE, "sim_runner.py"), "--queue", QUEUE, "--workers", W])
    time.sleep(60)
print("STOP_LOOP found - stopped")
