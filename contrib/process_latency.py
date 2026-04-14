import re
import sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_file>")
        sys.exit(1)

    filename = sys.argv[1]
    pattern = re.compile(r"!!LATENCY (S|E) id=(\d+) time=(\d+)")

    starts = {}
    ends = {}
    with open(filename, "r") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                typ, id_, time = match.groups()
                id_ = int(id_)
                time = int(time)

                if typ == "S":
                    starts[id_] = time
                elif typ == "E":
                    ends[id_] = time

    latencies_ms = []
    for id_ in starts:
        if id_ in ends:
            latency_ns = ends[id_] - starts[id_]
            latency_ms = latency_ns / 1_000_000
            latencies_ms.append(latency_ms)
    if not latencies_ms:
        print("No valid latency pairs found.")
        sys.exit(1)

    latencies_ms = np.array(latencies_ms)
    mean_latency = np.mean(latencies_ms)
    std_latency = np.std(latencies_ms)
    print("Latencies (ms):\n", latencies_ms)
    print("Mean latency (ms):", mean_latency)
    print("Std deviation (ms):", std_latency)

if __name__ == "__main__":
    main()

