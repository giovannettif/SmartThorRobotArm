# Sends commands from processing laptop to the control laptop.

import zmq
import random
import sys
import time

def send_command(A, BC, D, X, Y, Z):
    port = "5556"
    if len(sys.argv) > 1:
        port =  sys.argv[1]
        int(port)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect("tcp://192.168.1.111:%s" % port)


    socket.send(b"")
    time.sleep(1)

    # Base Rotation = A
    # A = 10
    # Shoulder - B & C
    # BC = 70
    # Elbow - D
    # D = 77
    # Wrist Rotation - X
    # X = 10
    # Hand Up and Down and Rotation,
    # Hand Counter Clockwise 90 Degrees: Y12.0 Z12.0
    # Hand Counter CounterClockwise 90 Degrees: Y-12.0 Z-12.0
    #
    # When hand commands are opposite such as, Y-9 Z9, hand moves up to other side (4.2 Hand faces straight up)
    # 
    # Y = 2
    # Z = 5

    command = f"G0 A{A} B{BC} C{BC} D{D} X{X} Y{Y} Z{Z}"

    print("Command:", command)

    socket.send(command.encode())


if __name__ == "__main__":
    send_command(13, 80, 57, 0, 2, 6) # Pre-calculated for picking up highlighter

    time.sleep(10)

    send_command(40, 80, 0, 0, 2, 6) # Pre-calculated for picking up highlighter