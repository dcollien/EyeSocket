import socket

#from pythonosc import osc_message_builder
#from pythonosc import udp_client

FORMAT = 'udp'

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

client = None
if FORMAT == 'udp':
	client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet, UDP

def send_data(data):
	if FORMAT == 'udp':
		client.sendto(data.encode('UTF-8'), (UDP_IP, UDP_PORT))
	else:
		print(data.encode('UTF-8'))
