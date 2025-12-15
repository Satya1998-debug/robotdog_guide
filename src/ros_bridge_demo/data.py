import base64
import cv2
import numpy as np
from roslibpy import Ros, Topic
from typing import TypedDict, Dict, Optional, Literal

RGB_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
RGB_ROS_MSG_TYPE = 'sensor_msgs/Image'
DEPTH_ROS_MSG_TYPE = 'sensor_msgs/Image'

latest_frame_color = None
latest_frame_depth = None

# image processing function
def decode_raw_image(msg):
    encoding = msg.get('encoding', '')
    height = msg.get('height', 0)
    width = msg.get('width', 0)
    data = msg.get('data', b'')
    
    if isinstance(data, str): # roslibpy sends data as base64 string
        data = base64.b64decode(data)  # convert base64 string to bytes
    
    print(f"Decoding image - Encoding: {encoding}, Size: {width}x{height}, Data length: {len(data)}")
    
    if encoding == 'rgb8':
        # RGB8: 3 bytes per pixel
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    elif encoding == 'bgr8':
        # BGR8: 3 bytes per pixel (OpenCV native format)
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        img = img_array
    elif encoding == 'mono8':
        # Mono8: 1 byte per pixel
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
    elif encoding == '16UC1':
        # 16-bit unsigned (typical for depth)
        img = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
    else:
        print(f"Warning: Unsupported encoding '{encoding}', attempting generic decode")
        img = np.frombuffer(data, dtype=np.uint8)
        if len(img) == height * width * 3:
            img = img.reshape((height, width, 3))
        elif len(img) == height * width:
            img = img.reshape((height, width))
    
    return img

def image_callback_color(msg):
    global latest_frame_color
    try:
        img = decode_raw_image(msg)
        latest_frame_color = img
        # print(f"Color image received - Shape: {img.shape}, Encoding: {msg.get('encoding', 'unknown')}")
    except Exception as e:
        print(f"Error decoding color image: {e}")

def image_callback_depth(msg):
    global latest_frame_depth
    try:
        img_depth = decode_raw_image(msg)
        latest_frame_depth = img_depth
        # print(f"Depth image received - Shape: {img_depth.shape}, Encoding: {msg.get('encoding', 'unknown')}, Dtype: {img_depth.dtype}")
    except Exception as e:
        print(f"Error decoding depth image: {e}")

def create_subscriber():
    try:
        # ros connection setup
        ros = Ros('localhost', 9090)
        ros.run()
        print("Connected to ROS:", ros.is_connected)

        # subscribe to image topics
        image_topic = Topic(ros, RGB_TOPIC, RGB_ROS_MSG_TYPE)
        depth_image_topic = Topic(ros, DEPTH_TOPIC, DEPTH_ROS_MSG_TYPE)
        image_topic.subscribe(image_callback_color)
        depth_image_topic.subscribe(image_callback_depth)

    except Exception as e:
        print(f"Error in creating ROS image subscriber creation: {e}")

def get_image_frames():
    global latest_frame_color, latest_frame_depth

    print("Fetching latest image frames for color and depth.")
    frame_color = latest_frame_color
    frame_depth = latest_frame_depth
    if frame_color is not None and frame_depth is not None:
        print("Returning latest image frames with shapes:", frame_color.shape, frame_depth.shape)
        return frame_color, frame_depth
    else:
        print("No image frames available yet.")
    return None, None