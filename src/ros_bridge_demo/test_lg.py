import base64
import cv2
import numpy as np
from roslibpy import Ros, Topic
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Optional, Literal

class robotState(TypedDict):
    counter: int

# ros connection setup
ros = Ros('localhost', 9090)
ros.run()

print("Connected to ROS:", ros.is_connected)

# subscribe to image topic
image_topic = Topic(ros, '/camera/color/image_raw', 'sensor_msgs/Image')
depth_image_topic = Topic(ros, '/camera/aligned_depth_to_color/image_raw', 'sensor_msgs/Image')
latest_frame_color = None
latest_frame_depth = None

# image processing function
def decode_raw_image(msg):
    """Decode raw ROS Image message to numpy array"""
    encoding = msg.get('encoding', '')
    height = msg.get('height', 0)
    width = msg.get('width', 0)
    data = msg.get('data', b'')
    
    # Handle base64-encoded data (roslibpy sends data as base64 string)
    if isinstance(data, str):
        data = base64.b64decode(data)
    
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

image_topic.subscribe(image_callback_color)
depth_image_topic.subscribe(image_callback_depth)


# define state graph
graph = StateGraph(robotState)

def get_image_node(robot_state: robotState) -> robotState:
    global latest_frame_color, latest_frame_depth
    print("\n=== Fetching latest image frame ===")
    frame_color = latest_frame_color
    frame_depth = latest_frame_depth
    counter = robot_state.get("counter", 0)
    counter += 1
    
    if frame_color is not None and frame_depth is not None:
        print(f"Color image - Shape: {frame_color.shape}, Dtype: {frame_color.dtype}")
        print(f"Depth image - Shape: {frame_depth.shape}, Dtype: {frame_depth.dtype}")
        
        # Save color image
        color_filename = f"latest_image_color_{counter}.jpg"
        cv2.imwrite(color_filename, frame_color)
        print(f"Saved color image: {color_filename}")
        
        # Save depth image (normalize for visualization if 16-bit)
        depth_filename = f"latest_image_depth_{counter}.png"
        if frame_depth.dtype == np.uint16:
            # Normalize 16-bit depth to 8-bit for visualization
            depth_normalized = cv2.normalize(frame_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(depth_filename, depth_normalized)
            # Also save raw depth data
            np.save(f"latest_image_depth_{counter}_raw.npy", frame_depth)
            print(f"Saved depth image: {depth_filename} (normalized) and .npy (raw)")
        else:
            cv2.imwrite(depth_filename, frame_depth)
            print(f"Saved depth image: {depth_filename}")
    else:
        print("No image frame available yet.")
    return {"counter": counter}

def should_continue(robot_state: robotState) -> Literal["image_node", END]:
    # Continue indefinitely for this example
    if robot_state["counter"] >= 10:
        print("Reached maximum count, ending graph.")
        return END
    print("Continuing to get more images...")
    time.sleep(5)  # wait for 5 seconds before next image
    return "image_node"

# define nodes
graph.add_node("image_node", get_image_node)

# define edges
graph.add_edge(START, "image_node")
graph.add_conditional_edges("image_node", should_continue)  # loop to get continuous images

# compute the graph
app = graph.compile()
print("Graph computation result:", app)


if __name__ == "__main__":
    print("starting graph...")
    import time
    try:    
        app.invoke({})
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        image_topic.unsubscribe()
        ros.terminate()
