import os
import subprocess
import sys


def main():
    """
    Launch 30 separate Python processes, one for each (K, H) configuration,
    and save their stdout/stderr to individual log files under ./log.
    This is equivalent to running 30 separate Python files in parallel.
    """
    Ks = [10, 50, 100]
    H_list = list(range(1, 11))

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    procs = []

    for K in Ks:
        for H in H_list:
            log_path = os.path.join(log_dir, f"ttt_K{K}_H{H}.log")
            cmd = [
                sys.executable,
                "ttt_run_config.py",
                "--K",
                str(K),
                "--H",
                str(H),
            ]
            print(f"Launching: {cmd} -> {log_path}")
            log_f = open(log_path, "wb")
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            procs.append((p, log_f))

    # Wait for all jobs to finish
    exit_codes = []
    for p, log_f in procs:
        code = p.wait()
        log_f.close()
        exit_codes.append(code)

    # Simple summary
    if all(code == 0 for code in exit_codes):
        print("All TTT jobs finished successfully.")
    else:
        print("Some TTT jobs exited with non-zero status:", exit_codes)


if __name__ == "__main__":
    main()

