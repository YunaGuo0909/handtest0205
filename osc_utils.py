# OSC: send gesture commands to UE. Main message: /hand/move (float) 1.0=forward, -1.0=back, 0.0=stop.
# Optional: /hand/debug (string) for debug.

from pythonosc import udp_client, dispatcher, osc_server


class OSCSender:
    """Send gesture commands to UE via OSC."""

    def __init__(self, ip, port):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.ip = ip
        self.port = port

    def send(self, move_value=0.0, command="none", gesture="unknown",
             palm_facing=False, num_hands=0):
        """Send single /hand/move message."""
        self.client.send_message("/hand/move", move_value)

    def close(self):
        pass


class OSCReceiver:
    """OSC receiver for testing (simulate UE receiving)."""

    COMMAND_LABELS = {
        1.0:  "Forward >>",
        -1.0: "<< Back",
        0.0:  "|| Stop",
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
            print(f"\n===== Command: {label} (move={value}) =====")
            self._last_value = value

        print(f"[{self._count}] move={value:+.1f}  {self._debug_info}")
        self._count += 1

    def _on_debug(self, address, value):
        self._debug_info = value

    def listen(self):
        """Listen for OSC messages. Ctrl+C to exit."""
        server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), self._dispatcher
        )
        print(f"OSC listening {self.ip}:{self.port}, Ctrl+C to quit...")
        print("Waiting for /hand/move...\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(f"\nQuit. Received {self._count} frames")
            server.shutdown()
