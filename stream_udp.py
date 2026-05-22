#!/usr/bin/env python3
"""
================================================================================
Host PC UDP Video Streamer for NuMaker-X-M55M1D Edge AI People Counting
================================================================================
This script captures video from a local webcam or file, processes it, and
streams raw grayscale frames over the network to the M55M1 board using a robust,
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
CHUNK_SIZE = 1024      # Size of the payload chunk in bytes
IMAGE_W = 192
IMAGE_H = 192

def stream_video(target_ip, target_port, source, fps, channels=3, chunk_delay=0.0005, bind_ip=""):
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
    
    # Initialize video capture (0 for default webcam, or string path to video file)
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[-] Error: Could not open video source: {source}")
        return

    frame_size = IMAGE_W * IMAGE_H * channels
    mode_str = "RGB" if channels == 3 else "Grayscale"
    print(f"[+] Starting video stream to {target_ip}:{target_port}...")
    print(f"[+] Capture resolution: {IMAGE_W}x{IMAGE_H} ({mode_str})")
    print(f"[+] Target Frame Rate: {fps} FPS")
    print("[+] Press 'q' in the OpenCV window to exit.")

    frame_id = 0
    delay = 1.0 / fps

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                # Loop back if video file ends
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[-] Failed to capture frame from camera.")
                    break

            # 1. Preprocess: Resize to 192x192
            resized = cv2.resize(frame, (IMAGE_W, IMAGE_H))
            
            # 2. Format conversion
            if channels == 3:
                # Convert BGR to RGB for model input
                processed_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                # Convert BGR to Grayscale
                processed_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # 3. Get raw byte array
            raw_bytes = processed_img.tobytes()
            assert len(raw_bytes) == frame_size, f"Invalid frame size: {len(raw_bytes)} (expected {frame_size})"

            # 4. Stream frame in chunks
            num_chunks = (frame_size + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            for i in range(num_chunks):
                offset = i * CHUNK_SIZE
                chunk_payload = raw_bytes[offset:offset + CHUNK_SIZE]
                chunk_len = len(chunk_payload)
                
                # Assemble Header (20 bytes):
                # magic (4s), frame_id (I), total_len (I), chunk_offset (I), chunk_len (I)
                header = struct.pack(
                    "!4sIIII",
                    MAGIC_HEADER,
                    frame_id,
                    frame_size,
                    offset,
                    chunk_len
                )
                
                # Send header + payload chunk
                sock.send(header + chunk_payload)
                if chunk_delay > 0:
                    time.sleep(chunk_delay)
            
            frame_id += 1
            
            # Local visualization (display in standard BGR for OpenCV window)
            display_img = resized if channels == 3 else processed_img
            cv2.imshow(f"Streaming to M55M1 ({mode_str} Feed)", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Cap FPS
            elapsed = time.time() - start_time
            if elapsed < delay:
                time.sleep(delay - elapsed)

    except KeyboardInterrupt:
        print("\n[+] Stream stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        print("[+] Stream closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host UDP Streamer for M55M1 People Counting")
    parser.add_argument("--ip", type=str, default="192.168.1.10", help="Target M55M1 board IP address")
    parser.add_argument("--port", type=int, default=5005, help="Target UDP port")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (e.g. '0') or path to video file")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second to stream")
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3], help="Number of image channels: 1 (Grayscale), 3 (RGB)")
    parser.add_argument("--chunk-delay", type=float, default=0.0005, help="Delay between UDP chunks in seconds; use 0 to disable pacing")
    parser.add_argument("--bind-ip", type=str, default="", help="Local PC interface IP to send from, useful when Windows chooses the wrong NIC")
    
    args = parser.parse_args()
    stream_video(args.ip, args.port, args.source, args.fps, args.channels, args.chunk_delay, args.bind_ip)
