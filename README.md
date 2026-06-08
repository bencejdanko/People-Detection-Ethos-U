# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

Uses YOLOv8n to conduct people counting, and pushes results to an HTTP server over Wifi.

YOLOv8n runs 20FPS at 640x480 centered. Sends people counts to dashboard over Wi-Fi connection.

## Usage

Install necessary libraries:

```bash
git clone https://github.com/OpenNuvoton/ML_M55M1_SampleCode

git clone https://github.com/OpenNuvoton/M55M1BSP
```

Load `MODEL.TFL` into the root of the SD card.

Start the web server, and make sure it is accessible under config in `board_config.h`.

```bash
python web_server.py
```

## Additional Resources

* [NuMaker-X-M55M1D User Manual](https://www.nuvoton.com/export/resource-files/en-us--UM_NuMaker-X-M55M1D_EN_Rev1.01.pdf)
* [Nuvoton NuMicro ICP Programming Tool](https://www.nuvoton.com/tool-and-software/debugger-and-programmer/1-to-1-debugger-and-programmer/nu-link2-pro/?index=4)