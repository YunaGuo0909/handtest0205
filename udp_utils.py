# UDP 通信模块
# 提供手部追踪数据的发送和接收功能

import socket
import json
import time


class UDPSender:
    """UDP 数据发送器 - 向 UE 或其他接收端发送手部数据和指令"""

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, hands, command="none"):
        """发送手部数据和游戏指令

        Args:
            hands: 手部数据列表
            command: 游戏指令 ("forward"/"backward"/"stop"/"none")

        Returns:
            dict: 发送的完整数据包
        """
        data = {
            "timestamp": time.time(),
            "command": command,
            "num_hands": len(hands),
            "hands": hands,
        }
        self.sock.sendto(json.dumps(data).encode(), (self.ip, self.port))
        return data

    def close(self):
        """关闭套接字"""
        self.sock.close()


class UDPReceiver:
    """UDP 数据接收器 - 用于测试，模拟 UE 接收数据"""

    COMMAND_LABELS = {
        "forward": "前进 >>",
        "backward": "<< 后退",
        "stop": "|| 停止",
        "none": "-- 待命",
    }

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))

    def listen(self):
        """开始监听并打印接收到的数据，按 Ctrl+C 退出"""
        print(f"监听 {self.ip}:{self.port}，按 Ctrl+C 退出...")
        count = 0
        last_command = None
        try:
            while True:
                data, _ = self.sock.recvfrom(65535)
                d = json.loads(data)

                command = d.get("command", "none")
                cmd_label = self.COMMAND_LABELS.get(command, command)

                if d["num_hands"] > 0:
                    parts = []
                    for hand in d["hands"]:
                        hand_type = hand.get("hand_type", "?")
                        gesture = hand.get("gesture", "?")
                        facing = "正面" if hand.get("palm_facing") else "背面"
                        parts.append(f"{hand_type}:{gesture}({facing})")
                    hand_info = " | ".join(parts)
                else:
                    hand_info = "无手"

                if command != last_command:
                    print(f"\n===== 指令切换: {cmd_label} =====")
                    last_command = command

                print(f"[{count}] {cmd_label}  {hand_info}")
                count += 1

        except KeyboardInterrupt:
            print(f"\n退出，共收到 {count} 帧")
        finally:
            self.sock.close()
