from freenove_arm import FreenoveArmClient

OPEN = 70
CLOSE = 0

with FreenoveArmClient(host="10.149.65.232", port=5000, auto_enable=True, verbose=True) as arm:
    arm.set_servo(0, OPEN)
    arm.wait(3.0)
    arm.set_servo(0, CLOSE)
    arm.wait(3.0)
