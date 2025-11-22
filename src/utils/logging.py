import os
import time

class SimpleLogger:
    """
    Lightweight console + file logger.
    """

    def __init__(self, log_dir="./logs", name="log"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(log_dir, f"{name}_{timestamp}.txt")

        with open(self.log_path, "w") as f:
            f.write(f"[Logger] Start: {timestamp}\n")

    def log(self, msg: str):
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")


LOGGER = SimpleLogger()
def log(msg: str):
    LOGGER.log(msg)
