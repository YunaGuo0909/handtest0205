# OSC 接收测试 - 模拟 UE 接收 OSC 消息
# 运行: python osc_receiver.py

from config import OSC_IP, OSC_PORT
from osc_utils import OSCReceiver


if __name__ == "__main__":
    receiver = OSCReceiver(OSC_IP, OSC_PORT)
    receiver.listen()
