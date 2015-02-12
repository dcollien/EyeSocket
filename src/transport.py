import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet, UDP

def send_data(data):
	sock.sendto(data.encode('UTF-8'), (UDP_IP, UDP_PORT))
