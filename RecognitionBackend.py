import socketserver
import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
import threading
import socket
import time

import utils

class MotorControler():
    def __init__(self, addr): 
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = addr
        self.last_cmd = 1000
                
    def send(self,cmd):
        if type(cmd) == tuple or type == list:# auto control cmd
            location = cmd
            if location[1] == 0:
                cmd = 1000 # 搜寻状态
            else:
                cmd = location[1]
            if cmd == self.last_cmd:
                return 0
            self.last_cmd = cmd
        else: #manual control cmd
            cmd = cmd.upper()
            if cmd == 'STOP':
                cmd = 1002
            elif cmd == 'UP':
                cmd = 1003
            elif cmd == 'DOWN':
                cmd = 1004
            elif cmd == 'LEFT':
                cmd = 1005
            elif cmd ==  'RIGHT':
                cmd = 1006
            else:
                print('Command %s not found'%cmd)
                return 0
        print('send cmd: %d'%cmd)
        self.sock.sendto(str(cmd).encode(),self.addr)
        
def get_target_v1(bboxes):
    ## bbox: [row_min,col_min,row_max,col_min]
    row_centre = (bboxes[:,1]+bboxes[:,3])/2
    col_centre = (bboxes[:,0]+bboxes[:,2])/2    
    index = np.argmax(row_centre)
    return (int(np.round(row_centre[index])),
            int(np.round(col_centre[index]))), index

def get_target(bboxes):
    ## bbox: [row_min,col_min,row_max,col_min]
    row_centre = (bboxes[:,1]+bboxes[:,3])/2
    col_centre = (bboxes[:,0]+bboxes[:,2])/2
    dis = np.square(320-col_centre) + np.square(479-row_centre)
    index = np.argmin(dis)
    return (int(np.round(row_centre[index])),
            int(np.round(col_centre[index]))), index
    
    
class VideoStreamHandler(socketserver.StreamRequestHandler):
    # 控制指令传输
    pi_motor_addr = ('192.168.1.104',8888)
    motorcontroler = MotorControler(pi_motor_addr)
    ui_image_addr = ('localhost',9999)
    ui_control_addr = ('localhost',11111)
    # 加载目标检测模型
    IMAGE_H, IMAGE_W = 640, 480
    classes = ['rubbish']
    num_classes = len(classes)    
    cpu_nms_graph = tf.Graph()
    input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./pb/yolov3_tiny_mix_low_q_enhance_3k.pb",
                                                                ["Placeholder:0", "concat_9:0", "mul_6:0"])    
    sess = tf.Session(graph=cpu_nms_graph)
    # 
    drawing_tools = None
    image = None
    
    flag_autocontrol = False
    
    def handle(self):
        #开始接收实时图像
        self.start_threads()
        stream_bytes = b' '
        while True:
            stream_bytes += self.rfile.read(1024)
            first = stream_bytes.find(b'\xff\xd8')
            last = stream_bytes.find(b'\xff\xd9')
            if first != -1 and last != -1:
                # 获取船只视野
                jpg = stream_bytes[first:last + 2]
                stream_bytes = stream_bytes[last + 2:]
                self.image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
    
    def start_threads(self):
        #启动图像识别线程
        detection_thread = threading.Thread(target=self.detection)
        detection_thread.daemon = True
        detection_thread.start()
        #启动电机控制线程
        control_thread = threading.Thread(target=self.autocontrol)
        control_thread.daemon = True
        control_thread.start()
        #启动检测结果展示线程
        sendresult_thread = threading.Thread(target=self.send_detection)
        sendresult_thread.daemon = True
        sendresult_thread.start()
        #命令监听线程
        cmdcolloct_thread = threading.Thread(target=self.cmdcollet)
        cmdcolloct_thread.daemon = True
        cmdcolloct_thread.start()
        
    def detection(self):
        while True:
            if self.image is not None:
                count = 0
                break
        while True:
            image = self.image
            image_ = ((np.array(image)/255.)-0.5)*2.
            boxes, scores = self.sess.run(self.output_tensors,feed_dict={self.input_tensor:np.expand_dims(image_, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, self.num_classes, score_thresh=0.3, iou_thresh=0.3)
            self.drawing_tools = {'id':count,'image':image,'boxes':boxes,
                                  'scores':scores,'labels':labels}
            count += 1
    
    def cmdcollet(self):
        ui_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ui_control_socket.bind(self.ui_control_addr)    
        while True:
            data, _ = ui_control_socket.recvfrom(32)
            cmd = data.decode()
            print('get cmd: %s'%cmd)
            if cmd.upper() == 'AUTO':
                self.flag_autocontrol = True
            elif cmd.upper() == 'MANUAL':
                self.flag_autocontrol = False
            else:
                self.motorcontroler.send(cmd)
               
    def autocontrol(self):
        while True:
            if self.flag_autocontrol:
                try:
                    boxes = self.drawing_tools['boxes']
                except TypeError as err:
                    print('未开始检测数据:', err)
                    time.sleep(0.5)
                    continue
                if boxes is None:
                    location = (0,0)
                    self.motorcontroler.send(location)
                else:
                    location, index = get_target(boxes)
                    self.motorcontroler.send(location)
                    time.sleep(0.05)
            else:
                time.sleep(1)
                
    def send_detection(self):
        while True:
            if self.image is not None:
                break
            time.sleep(0.5)
        # set socket
        ui_image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ui_image_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),75]
        # start sending image data
        while True:
            try:
                boxes = self.drawing_tools['boxes']
                if boxes is None:
                    image  = self.image
                else:
                    image, scores, labels = self.drawing_tools['image'], self.drawing_tools['scores'],\
                                            self.drawing_tools['labels']
                    _, index = get_target(boxes)
                    image = np.array(utils.draw_boxes(Image.fromarray(image),boxes,scores,labels,self.classes,
                                                       [self.IMAGE_H, self.IMAGE_W],target_index=index))
            except:
                image = self.image
            image = cv2.imencode('.jpeg',image,encode_param)[1]
            image = image.tostring()
            ui_image_socket.sendto(image, self.ui_image_addr)        
                        

class Server(object):
    def __init__(self, host, port1):
        self.host = host
        self.port1 = port1

    def video_stream(self, host, port):
        s = socketserver.TCPServer((host, port), VideoStreamHandler)
        s.serve_forever()

    def start(self):
        self.video_stream(self.host, self.port1)


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
    server_addr = ('',6666)
    print('Waiting Img at %s:%d'%(server_addr[0],server_addr[1]))
    ts = Server(server_addr[0], server_addr[1])
    ts.start()
