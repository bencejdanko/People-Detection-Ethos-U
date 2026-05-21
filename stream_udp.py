#!/usr/bin/env python3
"""
================================================================================
Host PC UDP Video Streamer for NuMaker-X-M55M1D Edge AI People Counting
================================================================================
This script captures video from a local webcam or file, processes it, and
streams raw grayscale frames over the network to the M55M1 board using a robust,
custom UDP chunking protocol.

Design Decision: Raw Grayscale Streaming
---------------------------------------
Instead of streaming JPEG which incurs CPU decompression overhead on the micro,
we stream raw grayscale (192x192 = 36,864 bytes) divided into 36 chunks of 1024
bytes. This is extremely lightweight, requires zero decompression cycles on the
Cortex-M55, and is highly robust against packet drops.
================================================================================
"""

import cv2
import socket
import struct
import time
import argparse

# Protocol Specifications
MAGIC_HEADER = b"FRME"  # 0x46524D45 in ASCII
CHUNK_SIZE = 1024      # Size of the payload chunk in bytes
IMAGE_W = 192
IMAGE_H = 192
FRAME_SIZE = IMAGE_W * IMAGE_H  # 36,864 bytes for raw 192x192 grayscale

def stream_video(target_ip, target_port, source, fps):
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Initialize video capture (0 for default webcam, or string path to video file)
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[-] Error: Could not open video source: {source}")
        return

    print(f"[+] Starting video stream to {target_ip}:{target_port}...")
    print(f"[+] Capture resolution: {IMAGE_W}x{IMAGE_H} (Grayscale)")
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
            
            # 2. Convert to Grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # 3. Get raw byte array
            raw_bytes = gray.tobytes()
            assert len(raw_bytes) == FRAME_SIZE, f"Invalid frame size: {len(raw_bytes)}"

            # 4. Stream frame in chunks
            num_chunks = (FRAME_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
            
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
                    FRAME_SIZE,
                    offset,
                    chunk_len
                )
                
                # Send header + payload chunk
                sock.sendto(header + chunk_payload, (target_ip, target_port))
            
            frame_id += 1
            
            # Local visualization
            cv2.imshow("Streaming to M55M1 (Grayscale Feed)", gray)
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
    
    args = parser.parse_args()
    stream_video(args.ip, args.port, args.source, args.fps)
