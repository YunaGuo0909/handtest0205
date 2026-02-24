# UDP 接收测试 - 模拟 UE 接收数据
# 运行: python udp_receiver.py

from config import UDP_IP, UDP_PORT
from udp_utils import UDPReceiver


if __name__ == "__main__":
    receiver = UDPReceiver(UDP_IP, UDP_PORT)
    receiver.listen()
