import threading
import time

for i in range(100):
    print(f"{i}", end='')
    time.sleep(1)
    print('\r', end='')
    time.sleep(1)
