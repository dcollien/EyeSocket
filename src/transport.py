import socket

from pythonosc import osc_message_builder
from pythonosc import udp_client

FORMAT = 'udp'

UDP_IP = "127.0.0.1"
UDP_PORT = 5005


if FORMAT == 'udp':
	client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet, UDP
elif FORMAT == 'osc':
	# OSC
	client = udp_client.UDPClient(UDP_IP, UDP_PORT)

def send_data(data):
	if FORMAT == 'udp':
		client.sendto(data.encode('UTF-8'), (UDP_IP, UDP_PORT))
	elif FORMAT == 'osc':
		msg = osc_message_builder.OscMessageBuilder(address = "/vision")
		msg.add_arg(data)
		msg = msg.build()
		client.send(msg)
	else:
		print(data.encode('UTF-8'))
