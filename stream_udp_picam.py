#!/usr/bin/env python3
"""
================================================================================
Host PC UDP Video Streamer for NuMaker-X-M55M1D Edge AI People Counting
(Modified for headless Raspberry Pi with Picamera2/libcamera support)
================================================================================
This script captures video from a local webcam or file, processes it, and
streams raw RGB/grayscale frames over the network to the M55M1 board using a robust,
custom UDP chunking protocol.
================================================================================
"""

import cv2
import socket
import struct
import time
import argparse
import ipaddress

# Protocol Specifications
MAGIC_HEADER = b"FRME"  # 0x46524D45 in ASCII
DEFAULT_CHUNK_SIZE = 1400  # Safe for 1500-byte Ethernet MTU with UDP/app headers
IMAGE_W = 192
IMAGE_H = 192

class OpenCVCapture:
    def __init__(self, cap, is_camera):
        self.cap = cap
        self.is_camera = is_camera
        self.color_order = "BGR"

    def read(self):
        return self.cap.read()

    def is_opened(self):
        return self.cap.isOpened()

    def restart(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        self.cap.release()


class Picamera2Capture:
    def __init__(self, picam2):
        self.picam2 = picam2
        self.is_camera = True
        self.color_order = "RGB"

    def read(self):
        return True, self.picam2.capture_array()

    def is_opened(self):
        return True

    def restart(self):
        pass

    def release(self):
        self.picam2.stop()
        self.picam2.close()


def open_picamera2_capture(fps):
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("[!] Picamera2 is not installed. Falling back to OpenCV/GStreamer.")
        return None

    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (IMAGE_W, IMAGE_H), "format": "RGB888"},
            buffer_count=3,
        )
        picam2.configure(config)
        frame_time_us = int(1_000_000 / fps)
        picam2.set_controls({"FrameDurationLimits": (frame_time_us, frame_time_us)})
        picam2.start()
        print("[+] Initialized Picamera2 at 192x192 RGB888.")
        return Picamera2Capture(picam2)
    except Exception as exc:
        print(f"[!] Picamera2 initialization failed: {exc}")
        return None


def open_opencv_capture(source, fps):
    if source == "0" or source == 0:
        gst_pipeline = (
            "libcamerasrc ! "
            f"video/x-raw,width={IMAGE_W},height={IMAGE_H},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink max-buffers=1 drop=true sync=false"
        )
        print("[+] Initializing libcamera via 192x192 GStreamer pipeline...")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("[!] GStreamer pipeline failed. Falling back to standard V4L2...")
            cap = cv2.VideoCapture(0)

        return OpenCVCapture(cap, is_camera=True)

    if isinstance(source, str) and source.isdigit():
        source = int(source)
    return OpenCVCapture(cv2.VideoCapture(source), is_camera=False)


def open_capture(source, fps, prefer_picamera2=True):
    if prefer_picamera2 and (source == "0" or source == 0):
        capture = open_picamera2_capture(fps)
        if capture is not None:
            return capture

    return open_opencv_capture(source, fps)


def prepare_frame(frame, color_order, channels):
    if frame.shape[1] != IMAGE_W or frame.shape[0] != IMAGE_H:
        frame = cv2.resize(frame, (IMAGE_W, IMAGE_H))

    if channels == 3:
        if color_order == "RGB":
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if color_order == "RGB":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def stream_video(target_ip, target_port, source, fps, channels=3, chunk_delay=0.0, bind_ip="", chunk_size=DEFAULT_CHUNK_SIZE):
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if bind_ip:
        sock.bind((bind_ip, 0))
    sock.connect((target_ip, target_port))
    local_ip, local_port = sock.getsockname()
    print(f"[+] UDP route: {local_ip}:{local_port} -> {target_ip}:{target_port}")
    try:
        local_addr = ipaddress.ip_address(local_ip)
        target_addr = ipaddress.ip_address(target_ip)
        if local_addr.version == 4 and target_addr.version == 4:
            same_24 = ipaddress.ip_network(f"{local_ip}/24", strict=False)
            if target_addr not in same_24:
                print(f"[!] Target is not in the sender's /24 subnet ({same_24}). Check the selected NIC or use --bind-ip.")
    except ValueError:
        pass

    capture = open_capture(source, fps)

    if not capture.is_opened():
        print(f"[-] Error: Could not open video source: {source}")
        print("[-] Tip: Install Picamera2, or try running the OpenCV fallback with: libcamerify python3 stream_udp_picam.py")
        capture.release()
        sock.close()
        return

    frame_size = IMAGE_W * IMAGE_H * channels
    mode_str = "RGB" if channels == 3 else "Grayscale"
    print(f"[+] Starting video stream to {target_ip}:{target_port}...")
    print(f"[+] Capture resolution: {IMAGE_W}x{IMAGE_H} ({mode_str})")
    print(f"[+] Target Frame Rate: {fps} FPS")
    print("[+] Running in headless mode (no GUI). Press Ctrl+C to exit.")

    frame_id = 0
    delay = 1.0 / fps

    try:
        while True:
            start_time = time.time()
            ret, frame = capture.read()
            if not ret:
                if not capture.is_camera:
                    capture.restart()
                    continue
                print("[-] Failed to capture frame from camera.")
                break

            processed_img = prepare_frame(frame, capture.color_order, channels)

            raw_bytes = processed_img.tobytes()
            assert len(raw_bytes) == frame_size, f"Invalid frame size: {len(raw_bytes)} (expected {frame_size})"

            num_chunks = (frame_size + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                offset = i * chunk_size
                chunk_payload = raw_bytes[offset:offset + chunk_size]
                chunk_len = len(chunk_payload)

                header = struct.pack(
                    "!4sIIII",
                    MAGIC_HEADER,
                    frame_id,
                    frame_size,
                    offset,
                    chunk_len
                )

                sock.send(header + chunk_payload)
                if chunk_delay > 0:
                    time.sleep(chunk_delay)

            # Console heartbeat instead of cv2.imshow
            if frame_id % (fps * 2) == 0:  # Print every 2 seconds worth of frames
                print(f"[Stream] Successfully dispatched frame {frame_id}...")

            frame_id += 1

            # Cap FPS
            elapsed = time.time() - start_time
            if elapsed < delay:
                time.sleep(delay - elapsed)

    except KeyboardInterrupt:
        print("\n[+] Stream stopped by user (Ctrl+C).")
    finally:
        capture.release()
        sock.close()
        print("[+] Stream closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host UDP Streamer for M55M1 People Counting")
    parser.add_argument("--ip", type=str, default="192.168.0.50", help="Target M55M1 board IP address")
    parser.add_argument("--port", type=int, default=5005, help="Target UDP port")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (e.g. '0') or path to video file")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second to stream")
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3], help="Number of image channels: 1 (Grayscale), 3 (RGB)")
    parser.add_argument("--chunk-delay", type=float, default=0.0, help="Delay between UDP chunks in seconds; use 0 to disable pacing")
    parser.add_argument("--bind-ip", type=str, default="", help="Local PC interface IP to send from")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="UDP image payload bytes per packet")

    args = parser.parse_args()
    stream_video(args.ip, args.port, args.source, args.fps, args.channels, args.chunk_delay, args.bind_ip, args.chunk_size)
