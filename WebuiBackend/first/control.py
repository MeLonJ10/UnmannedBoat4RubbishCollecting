# -*- coding: utf-8 -*-
from django.http import HttpResponse
import json
import socket

def getdirection(request):
    global udp_socket, control_addr
    if request.method=="POST":
        direction = request.POST.get('direction')
        udp_socket.sendto(direction.encode(),control_addr)
        print("Success...\nDirection:" + direction)
        return HttpResponse(json.dumps({
            "status":1,
            "result":"Success",
            "direction":direction
        }))

def getmode(request):
    global udp_socket, control_addr
    if request.method=="POST":
        switch_data = request.POST.get('switch_data')
        print("Success...\nSwitch:" + switch_data)
        if switch_data.upper() == 'AUTO':
            udp_socket.sendto('AUTO'.encode(),control_addr)
        elif switch_data.upper() == 'MANUAL':
            udp_socket.sendto('MANUAL'.encode(),control_addr)
        else:
            print('Error mode')
        return HttpResponse(json.dumps({
            "status":1,
            "result":"Success",
            "direction":switch_data
        }))

global udp_socket, control_addr
control_addr = ('localhost',11111)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)