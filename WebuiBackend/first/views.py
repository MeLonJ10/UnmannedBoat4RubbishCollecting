# -*- coding: utf-8 -*-
from django.http import HttpResponse
from django.shortcuts import render
import json
import socket
import csv
import os

def index(request):
    return render(request,'index.html')

def decode_data(data):
    data = data[1:-1]
    data = data.split(', ')
    r_t = []
    for value in data:
        try:
            temp = int(value)
        except ValueError:
            temp = value[1:-1]
        r_t.append(temp)
    return r_t
    
#def data_list(request):
def getmotordata(request):
    global udp_socket, pi_addr, namelist
    udp_socket.sendto('Hust'.encode(),pi_addr)
    data = udp_socket.recv(1024).decode()
    data = decode_data(data)
    with open('./rec_gps.csv','a',encoding='utf-8',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
        csvFile.close()
    data_dict = {}
    for i, _data in enumerate(data):
        data_dict.update({namelist[i]:_data})
    return HttpResponse(json.dumps(data_dict), content_type='application/json')

global namelist
namelist = ('waterTemp','waterTurb','waterLevel','airHumidity','gpstime',
            'longitude','latitude','altitude','motorLeft','motorRight',
            'batteryLeft','batteryRight')
if not os.path.exists('./rec_gps.csv'): 
    with open('./rec_gps.csv','w',encoding='utf-8',newline='') as csvFile:
        writer=csv.writer(csvFile)
        writer.writerow(namelist)
        csvFile.close()
global udp_socket, pi_addr
pi_addr = ('localhost',7777)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

