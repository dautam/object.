
import socket
import time
#from djitellopy import Tello
#we can use tello instead socket if we want.
# Connect to the drone and send takeoff command
host = ''
port = 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

print("Send command")
sock.sendto(b'command', ('192.168.10.1', 8889))
time.sleep(5)  # Wait for the drone to respond
print("Send takeoff")
sock.sendto(b'takeoff', ('192.168.10.1', 8889))
time.sleep(5)  # Wait for the drone to take off

# Send landing command
print("Send landing")
sock.sendto(b'land', ('192.168.10.1', 8889))
time.sleep(5)  # Wait for the drone to start landing

# Send emergency stop command (if necessary)
sock.sendto(b'emergency', ('192.168.10.1', 8889))
time.sleep(5)  # Wait for the drone to stop

# Close the socket
sock.close()