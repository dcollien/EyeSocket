import socket
import json
import pprint

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
   buf, addr = sock.recvfrom(1024) # buffer size is 1024 bytes

   json_str = str(buf, 'UTF-8')

   print("received:")
   try:
      data = json.loads(json_str)
   except ValueError as err:
      print(json_str, err)
   else:
      print(pprint.pprint(data))

