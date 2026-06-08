# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

Uses YOLOv8n to conduct people counting, and pushes results to an HTTP server over Wifi.

## Usage

Load `MODEL.TFL` into the root of the SD card.

Start the web server:
```bash
python web_server.py
```

## Additional Resources

* [NuMaker-X-M55M1D User Manual](https://www.nuvoton.com/export/resource-files/en-us--UM_NuMaker-X-M55M1D_EN_Rev1.01.pdf)
* [Nuvoton NuMicro ICP Programming Tool](https://www.nuvoton.com/tool-and-software/debugger-and-programmer/1-to-1-debugger-and-programmer/nu-link2-pro/?index=4)