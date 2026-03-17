# OSC receiver test - simulate UE receiving OSC. Run: python osc_receiver.py

from config import OSC_IP, OSC_PORT
from osc_utils import OSCReceiver


if __name__ == "__main__":
    receiver = OSCReceiver(OSC_IP, OSC_PORT)
    receiver.listen()
