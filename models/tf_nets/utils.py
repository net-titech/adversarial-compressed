import time

start = 0

def log_training(step, bloss):
    global start
    if start == 0:
        start = time.time()
        print("|\tStart| Step {}: batch loss: {}".format(step, bloss))
    else:
        elapsed = time.time() - start
        print("|\t{0:5.2f}(sec)| Step {1}: batch loss: {2}".format(elapsed, step, bloss))
