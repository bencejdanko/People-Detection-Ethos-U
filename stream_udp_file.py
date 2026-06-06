#!/usr/bin/env python3
"""
==============================================================================
Host PC UDP Video File Streamer for NuMaker-X-M55M1D Edge AI People Counting
==============================================================================
This script streams a local video file to the M55M1 board using the same raw
frame UDP chunking protocol as stream_udp.py.

Use this when you want to replay a recorded 192x192 video feed over the board.
"""

import argparse
from stream_udp import stream_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host UDP video file streamer for M55M1 People Counting")
    parser.add_argument("--ip", type=str, default="192.168.0.50", help="Target M55M1 board IP address")
    parser.add_argument("--port", type=int, default=5005, help="Target UDP port")
    parser.add_argument("--video-file", type=str, required=True, help="Path to the input MP4 video file")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second to stream")
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3], help="Number of image channels: 1 (Grayscale), 3 (RGB)")
    parser.add_argument("--chunk-delay", type=float, default=0.0005, help="Delay between UDP chunks in seconds; use 0 to disable pacing")
    parser.add_argument("--bind-ip", type=str, default="", help="Local PC interface IP to send from, useful when Windows chooses the wrong NIC")
    parser.add_argument("--chunk-size", type=int, default=1400, help="UDP image payload bytes per packet; keep <= 1400 to avoid fragmentation")
    parser.add_argument("--no-display", dest="show_window", action="store_false", help="Disable the OpenCV display window to reduce host overhead")
    parser.add_argument("--fast", action="store_true", help="Stream frames as fast as possible by skipping frame pacing")
    args = parser.parse_args()

    stream_video(
        args.ip,
        args.port,
        args.video_file,
        args.fps,
        args.channels,
        args.chunk_delay,
        args.bind_ip,
        args.chunk_size,
        show_window=args.show_window,
        fast=args.fast
    )
