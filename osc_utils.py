# OSC 通信模块
# 通过 OSC 协议向 UE 发送手势指令
#
# 核心消息（UE 读这个就够了）:
#   /hand/move  (float)  →  1.0=前进, -1.0=后退, 0.0=停止
#
# 调试消息（可选）:
#   /hand/debug (string) →  "forward|fist|F|1" 格式的调试信息

from pythonosc import udp_client, dispatcher, osc_server


class OSCSender:
    """OSC 发送器 - 向 UE 发送手势指令"""

    def __init__(self, ip, port):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.ip = ip
        self.port = port

    def send(self, move_value=0.0, command="none", gesture="unknown",
             palm_facing=False, num_hands=0):
        """发送 OSC 消息到 UE

        Args:
            move_value: 移动值 (1.0=前进, -1.0=后退, 0.0=停止)
            command: 指令名称（调试用）
            gesture: 手势名称（调试用）
            palm_facing: 手掌朝向（调试用）
            num_hands: 手的数量（调试用）
        """
        self.client.send_message("/hand/move", move_value)

        facing = "F" if palm_facing else "B"
        debug = f"{command}|{gesture}|{facing}|{num_hands}"
        self.client.send_message("/hand/debug", debug)

    def close(self):
        pass


class OSCReceiver:
    """OSC 接收器 - 用于测试，模拟 UE 接收 OSC 消息"""

    COMMAND_LABELS = {
        1.0:  "前进 >>",
        -1.0: "<< 后退",
        0.0:  "|| 停止",
    }

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self._count = 0
        self._last_value = None

        self._dispatcher = dispatcher.Dispatcher()
        self._dispatcher.map("/hand/move", self._on_move)
        self._dispatcher.map("/hand/debug", self._on_debug)

        self._debug_info = ""

    def _on_move(self, address, value):
        label = self.COMMAND_LABELS.get(value, f"? ({value})")

        if value != self._last_value:
            print(f"\n===== 指令切换: {label} (move={value}) =====")
            self._last_value = value

        print(f"[{self._count}] move={value:+.1f}  {self._debug_info}")
        self._count += 1

    def _on_debug(self, address, value):
        self._debug_info = value

    def listen(self):
        """开始监听 OSC 消息，按 Ctrl+C 退出"""
        server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), self._dispatcher
        )
        print(f"OSC 监听 {self.ip}:{self.port}，按 Ctrl+C 退出...")
        print(f"等待 /hand/move 消息...\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(f"\n退出，共收到 {self._count} 帧")
            server.shutdown()
