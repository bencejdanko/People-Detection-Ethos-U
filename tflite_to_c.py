#!/usr/bin/env python3
"""
tflite_to_c.py - Model serialization utility for Edge AI People Counting.
Converts a binary .tflite model file into a C/C++ header file with a raw byte array.
"""

import sys
import os

def convert_tflite_to_c(tflite_path, header_path):
    if not os.path.exists(tflite_path):
        print(f"Error: Source model file '{tflite_path}' not found!")
        sys.exit(1)
        
    print(f"Reading model file: {tflite_path}...")
    with open(tflite_path, 'rb') as f:
        data = f.read()
        
    file_size = len(data)
    print(f"Loaded {file_size} bytes.")
    
    print(f"Generating C/C++ header file: {header_path}...")
    with open(header_path, 'w') as f:
        f.write("/*\n")
        f.write(" * Auto-generated model byte array header.\n")
        f.write(f" * Generated from: {os.path.basename(tflite_path)}\n")
        f.write(f" * Size: {file_size} bytes\n")
        f.write(" */\n\n")
        f.write("#ifndef EMBEDDED_MODEL_H\n")
        f.write("#define EMBEDDED_MODEL_H\n\n")
        f.write("#ifdef __cplusplus\n")
        f.write("extern \"C\" {\n")
        f.write("#endif\n\n")
        
        f.write("/* Raw TFLite model data. Keep const so it stays in embedded flash. */\n")
        f.write("__attribute__((aligned(16)))\n")
        f.write("const unsigned char g_model_tflite[] = {\n")
        
        # Write bytes in chunks of 12 for clean layout
        for i, b in enumerate(data):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
                
        f.write("\n};\n\n")
        f.write(f"const unsigned int g_model_tflite_len = {file_size};\n\n")
        
        f.write("#ifdef __cplusplus\n")
        f.write("}\n")
        f.write("#endif\n\n")
        f.write("#endif /* EMBEDDED_MODEL_H */\n")
        
    print("Success! embedded_model.h generated successfully.")

if __name__ == "__main__":
    src = "model.tflite"
    dest = "embedded_model.h"
    
    if len(sys.argv) > 1:
        src = sys.argv[1]
    if len(sys.argv) > 2:
        dest = sys.argv[2]
        
    convert_tflite_to_c(src, dest)
