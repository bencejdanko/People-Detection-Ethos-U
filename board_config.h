/**************************************************************************//**
 * @file     board_config.h
 * @version  V1.00
 * @brief    System configurations for people counting alongside UDP server
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __BOARD_CONFIG_H__
#define __BOARD_CONFIG_H__

#include "NuMicro.h"

/* Board Model Configuration */
#define __NUMAKER_M55M1__

#if defined(__NUMAKER_M55M1__)
    #define __EBI_LCD_PANEL__
    #define CONFIG_LCD_EBI                  EBI_BANK0
    #define CONFIG_LCD_EBI_ADDR             (EBI_BANK0_BASE_ADDR+(CONFIG_LCD_EBI*EBI_MAX_SIZE))
    #define CONFIG_LCD_EBI_CLK_MODULE       EBI0_MODULE
    #define LT7381_LCD_PANEL
#endif

/* --- CAMERA SOURCE CONFIGURATION ---
 * Set USE_CCAP_CAMERA to 1 to source frames from the onboard HM1055 camera
 * via the CCAP connector.
 *
 * Set USE_CCAP_CAMERA to 0 to revert to a UDP network video feed.
 * A host PC must run stream_udp.py (or stream_udp_picam.py) to push RGB888
 * frames to UDP_STREAM_PORT on this device.
 */
#define USE_CCAP_CAMERA                1   // Toggle 1=CCAP onboard camera, 0=UDP network feed

/* --- SYSTEM LOGGING CONFIGURATION --- */
#define ENABLE_SERIAL_LOGS         1   // Toggle 1/0 to enable/disable serial logging
#define ENABLE_INFO_LOGS           1   // Toggle 1/0 to enable/disable detailed [INFO] logs
#define ENABLE_ERR_LOGS            1   // Toggle 1/0 to enable/disable [ERROR] logs

#if ENABLE_SERIAL_LOGS
    #define LOG_INFO(fmt, ...)   do { if(ENABLE_INFO_LOGS) printf("[INFO] " fmt "\n", ##__VA_ARGS__); } while(0)
    #define LOG_ERROR(fmt, ...)  do { if(ENABLE_ERR_LOGS)  printf("[ERROR] " fmt "\n", ##__VA_ARGS__); } while(0)
#else
    #define LOG_INFO(fmt, ...)   do {} while(0)
    #define LOG_ERROR(fmt, ...)  do {} while(0)
#endif

/* --- NETWORK CONFIGURATION --- */
#define LWIP_DHCP_ENABLE           1   // Set 1 for DHCP, 0 for static IP configuration

#if !LWIP_DHCP_ENABLE
    #define STATIC_IP_ADDR         "192.168.0.50"
    #define STATIC_NETMASK         "255.255.255.0"
    #define STATIC_GATEWAY         "192.168.0.1"
#endif

#define BOARD_MAC_ADDR             {0x00, 0x00, 0x24, 0xD4, 0x10, 0x30}

/* --- UDP VIDEO STREAM PROTOCOL CONFIGURATION --- */
#define UDP_STREAM_PORT            5005
#define UDP_MAGIC_HEADER           0x46524D45   // "FRME" in hex

/* Image size matching model input size */
#define IMAGE_WIDTH                192
#define IMAGE_HEIGHT               192
#define IMAGE_CHANNELS             3            // RGB (for full RGB implementation)
#define FRAME_BUFFER_SIZE          (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)

/* LCD preview size */
#if defined(LT7381_LCD_PANEL)
    #define LCD_DISPLAY_WIDTH      800
    #define LCD_DISPLAY_HEIGHT     480
#elif defined(FSA506_LCD_PANEL)
    #define LCD_DISPLAY_WIDTH      480
    #define LCD_DISPLAY_HEIGHT     272
#else
    #define LCD_DISPLAY_WIDTH      IMAGE_WIDTH
    #define LCD_DISPLAY_HEIGHT     IMAGE_HEIGHT
#endif
#define LCD_FRAME_BUFFER_SIZE      (LCD_DISPLAY_WIDTH * LCD_DISPLAY_HEIGHT * 2)

/* --- MODEL INFERENCE CONFIGURATION --- */
#define MODEL_INPUT_WIDTH          192
#define MODEL_INPUT_HEIGHT         192
#define MODEL_OUTPUT_GRID_SIZE     24          // 24x24 grid outputs from final Conv2D layer

/* Inference parameters */
#define MODEL_DEFAULT_THRESHOLD    0.50f       // Detection confidence threshold
#define MODEL_MIN_PEAK_DISTANCE    2.0f        // Grid-space NMS threshold (minimum distance between peaks)
#define MODEL_MAX_DETECTIONS       32          // Fixed result capacity; avoids heap allocation in post-processing

#endif // __BOARD_CONFIG_H__
