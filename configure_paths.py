#!/usr/bin/env python3
"""
================================================================================
Keil / CMSIS Project Path Configuration Utility
================================================================================
This utility configures the external Nuvoton SDK pathways ('Library' and 'ThirdParty')
inside the project files:
1. KEIL/PeopleCounting.uvprojx (Keil Project definition)
2. KEIL/PeopleCounting.csolution.yml (Arm CMSIS / csolution solution config)

This is ideal for VS Code MDK/CMSIS extensions, allowing direct compilation on 
Windows, Linux, or macOS.

Usage:
------
    python3 configure_paths.py --library "C:\\Library" --thirdparty "C:\\ThirdParty"
================================================================================
"""

import os
import re
import argparse

def configure_paths(library_path, thirdparty_path):
    # 1. Update Keil XML Project File
    uvproj_file = os.path.join("KEIL", "PeopleCounting.uvprojx")
    if os.path.exists(uvproj_file):
        print(f"[+] Loading Keil project: {uvproj_file}")
        with open(uvproj_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Normalize slashes for the MDK project XML schema
        new_lib = library_path.replace("/", "\\")
        new_tp = thirdparty_path.replace("/", "\\")

        targets = [
            ("..\\..\\..\\..\\Library", new_lib),
            ("$(NUVOTON_LIBRARY)", new_lib),
            ("..\\..\\..\\..\\ThirdParty", new_tp),
            ("$(NUVOTON_THIRDPARTY)", new_tp)
        ]

        modified = False
        for old, new in targets:
            count = content.count(old)
            if count > 0:
                content = content.replace(old, new)
                print(f"  -> Replaced {count} instances of '{old}' with '{new}'")
                modified = True

        if modified:
            with open(uvproj_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[+] {uvproj_file} successfully updated!")
    else:
        print(f"[-] Keil project file not found at: {uvproj_file} (skipping)")

    # 2. Update CMSIS csolution file
    csol_file = os.path.join("KEIL", "PeopleCounting.csolution.yml")
    if os.path.exists(csol_file):
        print(f"[+] Loading CMSIS csolution config: {csol_file}")
        with open(csol_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Slashes for csolution are typically forward slashes
        new_lib_fwd = library_path.replace("\\", "/")
        new_tp_fwd = thirdparty_path.replace("\\", "/")

        # Use regex to replace BSP_PATH and TP_PATH values dynamically
        content, count_bsp = re.subn(r'(BSP_PATH:\s*)"[^"]*"', r'\g<1>"{}"'.format(new_lib_fwd), content)
        content, count_tp = re.subn(r'(TP_PATH:\s*)"[^"]*"', r'\g<1>"{}"'.format(new_tp_fwd), content)

        if count_bsp > 0 or count_tp > 0:
            with open(csol_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  -> Updated BSP_PATH to \"{new_lib_fwd}\"")
            print(f"  -> Updated TP_PATH to \"{new_tp_fwd}\"")
            print(f"[+] {csol_file} successfully updated!")
    else:
        print(f"[-] CMSIS solution config not found at: {csol_file} (skipping)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure Keil / CMSIS project paths")
    parser.add_argument(
        "--library", 
        type=str, 
        required=True, 
        help="Path to Nuvoton 'Library' folder (e.g., C:\\Library)"
    )
    parser.add_argument(
        "--thirdparty", 
        type=str, 
        required=True, 
        help="Path to Nuvoton 'ThirdParty' folder (e.g., C:\\ThirdParty)"
    )

    args = parser.parse_args()
    configure_paths(args.library, args.thirdparty)
