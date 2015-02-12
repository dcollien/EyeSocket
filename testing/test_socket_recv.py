import socket
import json
import pprint

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

lastbuf = ''
while True:
    buf, addr = sock.recvfrom(1024) # buffer size is 1024 bytes

    data = str(buf, 'UTF-8')

    lastbuf += data

    lines = lastbuf.split('\n')

    if len(lines) > 1:
        for line in lines[:-1]:
            print(line)

    lastbuf = lines[-1]
