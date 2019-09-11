from django.http import HttpResponse
import socket
import threading

def getimgbuffer_forever():
    global img_buffer, listen_addr
    cli_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cli_socket.bind(listen_addr)
    while True:
        try:
            img_buffer, address = cli_socket.recvfrom(32768)
        except:
            print('Error while connecting with image-process server')
            continue

def getmotion(request):
    global img_buffer
    i = img_buffer
    return HttpResponse(i, content_type="image/jpg")


global img_buffer, listen_addr
listen_addr = ('',9999)
thred_recv = threading.Thread(target=getimgbuffer_forever)
thred_recv.daemon = True
thred_recv.start()