# OSC 通信模块
# 通过 OSC 协议向 UE 发送手势指令和追踪数据
#
# OSC 地址设计:
#   /hand/command    (string)  → 游戏指令: "forward" / "backward" / "stop" / "none"
#   /hand/gesture    (string)  → 手势名称: "fist" / "open" / ...
#   /hand/palm       (int)     → 手掌朝向: 1=正面(手心朝镜头) / 0=背面
#   /hand/hands      (int)     → 检测到的手数量
#   /hand/confidence (float)   → 主控手置信度

from pythonosc import udp_client, dispatcher, osc_server
import threading


class OSCSender:
    """OSC 发送器 - 向 UE 发送手势指令"""

    def __init__(self, ip, port):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.ip = ip
        self.port = port

    def send(self, command="none", gesture="unknown", palm_facing=False,
             num_hands=0, confidence=0.0):
        """发送 OSC 消息到 UE

        Args:
            command: 游戏指令
            gesture: 手势名称
            palm_facing: 手心是否朝镜头
            num_hands: 手的数量
            confidence: 主控手置信度
        """
        self.client.send_message("/hand/command", command)
        self.client.send_message("/hand/gesture", gesture)
        self.client.send_message("/hand/palm", 1 if palm_facing else 0)
        self.client.send_message("/hand/hands", num_hands)
        self.client.send_message("/hand/confidence", confidence)

    def close(self):
        """兼容接口，OSC client 无需显式关闭"""
        pass


class OSCReceiver:
    """OSC 接收器 - 用于测试，模拟 UE 接收 OSC 消息"""

    COMMAND_LABELS = {
        "forward": "前进 >>",
        "backward": "<< 后退",
        "stop": "|| 停止",
        "none": "-- 待命",
    }

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self._last_command = None
        self._count = 0

        self._dispatcher = dispatcher.Dispatcher()
        self._dispatcher.map("/hand/command", self._on_command)
        self._dispatcher.map("/hand/gesture", self._on_gesture)
        self._dispatcher.map("/hand/palm", self._on_palm)
        self._dispatcher.map("/hand/hands", self._on_hands)
        self._dispatcher.map("/hand/confidence", self._on_confidence)

        self._current_gesture = "?"
        self._current_palm = "?"
        self._current_hands = 0

    def _on_command(self, address, value):
        cmd_label = self.COMMAND_LABELS.get(value, value)

        if value != self._last_command:
            print(f"\n===== 指令切换: {cmd_label} =====")
            self._last_command = value

        palm_tag = "正面" if self._current_palm == 1 else "背面"
        print(
            f"[{self._count}] {cmd_label}  "
            f"手势:{self._current_gesture}({palm_tag})  "
            f"手数:{self._current_hands}"
        )
        self._count += 1

    def _on_gesture(self, address, value):
        self._current_gesture = value

    def _on_palm(self, address, value):
        self._current_palm = value

    def _on_hands(self, address, value):
        self._current_hands = value

    def _on_confidence(self, address, value):
        pass

    def listen(self):
        """开始监听 OSC 消息，按 Ctrl+C 退出"""
        server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), self._dispatcher
        )
        print(f"OSC 监听 {self.ip}:{self.port}，按 Ctrl+C 退出...")
        print(f"等待 /hand/command 消息...\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(f"\n退出，共收到 {self._count} 条指令")
            server.shutdown()
