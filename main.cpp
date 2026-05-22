/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    People counting alongside UDP video receiver server
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// Avoid ISO C++17 'register' storage class specifier errors in old Nuvoton SDK headers
#define register
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>

/* FreeRTOS includes */
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

/* LwIP network includes */
extern "C" {
#include "lwip/netifapi.h"
#include "lwip/tcpip.h"
#include "netif/ethernetif.h"
#include "lwip/api.h"
#include "lwip/def.h"
}

/* Board and BSP includes */
#include "NuMicro.h"
#include "BoardInit.hpp"
#include "board_config.h"
#include "pmu_counter.h"

/* Global variables and TrustZone stub symbols for non-secure FreeRTOS */
extern "C" {
    uint8_t my_mac_addr[6] = BOARD_MAC_ADDR;

    // FreeRTOS portasm.c unconditionally references these symbols for TrustZone stack context allocation,
    // but they are unused when configENABLE_TRUSTZONE is 0. We define dummies to satisfy the linker.
    void *xSecureContext = NULL;
    void SecureContext_SaveContext(void *xSecureContext, void *pxCurrentTCB) {}
    void SecureContext_LoadContext(void *xSecureContext, void *pxCurrentTCB) {}
}

/* Image Processing and OpenMV includes */
#include "imlib.h"
#include "framebuffer.h"

/* Model and ML includes */
#include "InferenceModel.hpp"
#include "PostProcessor.hpp"
#include "ModelFileReader.h"
#include "ff.h"

#if defined(__EBI_LCD_PANEL__)
#include "Display.h"
#endif

/* HyperRAM allocation address for model file */
#define MODEL_AT_HYPERRAM_ADDR    (0x82400000)

/* Task Handles */
static TaskHandle_t xUdpReceiverTaskHandle = NULL;
static TaskHandle_t xInferenceTaskHandle = NULL;

/* Network interface structure */
struct netif g_netif;

/* --- DOUBLE BUFFER MEMORY ALLOCATION --- */
// networkFrameBuffer is written to by UDP Receiver task
__attribute__((section(".bss.sram.data"), aligned(32))) static uint8_t g_networkFrameBuffer[FRAME_BUFFER_SIZE];
// inferenceFrameBuffer is read by ML Inference task
__attribute__((section(".bss.sram.data"), aligned(32))) static uint8_t g_inferenceFrameBuffer[FRAME_BUFFER_SIZE];

/* Tensor arena buffer for TensorFlow Lite Micro placed in SRAM01_HYPERRAM */
namespace arm {
namespace app {
__attribute__((section(".bss.NoInit.activation_buf_sram"), aligned(32))) static uint8_t tensorArena[ACTIVATION_BUF_SZ];
}
}

/* OpenMV Frame buffer memory allocations (re-used from YOLOv8n) */
#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE        (1*1024)
#define IMAGE_FB_SIZE            (320 * 240 * 2) // RGB565 320x240 for display
#undef OMV_FB_SIZE
#define OMV_FB_SIZE              (IMAGE_FB_SIZE + 1024)

__attribute__((section(".bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];
__attribute__((section(".bss.sram.data"), aligned(32))) static char frame_buf1[OMV_FB_SIZE];

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

/* Initialize OpenMV (imlib) frame buffer */
static void omv_init()
{
    image_t frameBuffer;
    frameBuffer.w = 320;
    frameBuffer.h = 240;
    frameBuffer.size = IMAGE_FB_SIZE;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();
    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);
}

/* UDP Frame Protocol Structure */
struct PacketHeader {
    uint32_t magic;         // Magic Header (0x46524D45 -> "FRME")
    uint32_t frame_id;      // Sequential Frame ID
    uint32_t total_len;     // Expected total size of raw pixels (36864)
    uint32_t chunk_offset;  // Offset of this chunk in the frame buffer
    uint32_t chunk_len;     // Length of this chunk payload
};

/* --- UDP Video Receiver Task --- */
static void vUdpVideoReceiverTask(void *pvParameters)
{
    (void)pvParameters;
    struct netconn *conn;
    err_t err;
    struct netbuf *buf;
    
    LOG_INFO("UDP Video Receiver Task started.");

    // Create a new UDP connection
    conn = netconn_new(NETCONN_UDP);
    if (conn == NULL) {
        LOG_ERROR("Failed to create UDP connection.");
        vTaskDelete(NULL);
        return;
    }

    // Bind to the configured port
    err = netconn_bind(conn, NULL, UDP_STREAM_PORT);
    if (err != ERR_OK) {
        LOG_ERROR("Failed to bind UDP connection to port %d.", UDP_STREAM_PORT);
        netconn_delete(conn);
        vTaskDelete(NULL);
        return;
    }

    LOG_INFO("UDP server listening on port %d...", UDP_STREAM_PORT);

    uint32_t activeFrameId = 0xFFFFFFFF;
    uint32_t accumulatedBytes = 0;

    while (1)
    {
        // Receive packet (blocks until data arrives)
        if (netconn_recv(conn, &buf) == ERR_OK)
        {
            uint16_t packetLen = buf->p->len;
            
            // Validate packet size (must contain at least the 20-byte header)
            if (packetLen >= sizeof(PacketHeader))
            {
                PacketHeader *header = (PacketHeader *)buf->p->payload;
                
                // Parse Network Byte Order (Big Endian) to Host Byte Order
                uint32_t magic = ntohl(header->magic);
                uint32_t frameId = ntohl(header->frame_id);
                uint32_t totalLen = ntohl(header->total_len);
                uint32_t chunkOffset = ntohl(header->chunk_offset);
                uint32_t chunkLen = ntohl(header->chunk_len);

                // 1. Verify Magic header and total length compatibility
                if (magic == UDP_MAGIC_HEADER && totalLen == FRAME_BUFFER_SIZE)
                {
                    // 2. Start a new frame if chunk_offset is 0
                    if (chunkOffset == 0)
                    {
                        activeFrameId = frameId;
                        accumulatedBytes = 0;
                    }

                    // 3. Write chunk to active frame buffer
                    if (frameId == activeFrameId && (chunkOffset + chunkLen) <= FRAME_BUFFER_SIZE)
                    {
                        uint8_t *payloadData = (uint8_t *)buf->p->payload + sizeof(PacketHeader);
                        memcpy(g_networkFrameBuffer + chunkOffset, payloadData, chunkLen);
                        accumulatedBytes += chunkLen;

                        // 4. Frame complete check
                        if (accumulatedBytes == FRAME_BUFFER_SIZE)
                        {
                            // Copy to Inference Frame Buffer
                            memcpy(g_inferenceFrameBuffer, g_networkFrameBuffer, FRAME_BUFFER_SIZE);
                            
                            // Notify the inference task
                            if (xInferenceTaskHandle != NULL)
                            {
                                xTaskNotifyGive(xInferenceTaskHandle);
                            }
                        }
                    }
                }
            }
            netbuf_delete(buf);
        }
    }
}

/* Copy model file from SD Card to HyperRAM */
static int32_t LoadModelFromSDCard(void)
{
#define MODEL_FILE_PATH "0:\\model.tflite"
#define EACH_READ_SIZE 512
    
    TCHAR sd_path[] = { '0', ':', 0 };
    f_chdrive(sd_path);

    int32_t fileSize;
    int32_t fileReadIndex = 0;
    int32_t bytesRead;
    
    LOG_INFO("Opening model file: %s", MODEL_FILE_PATH);
    if (!ModelFileReader_Initialize(MODEL_FILE_PATH))
    {
        LOG_ERROR("Unable to open model: %s", MODEL_FILE_PATH);        
        return -1;
    }
    
    fileSize = ModelFileReader_FileSize();
    LOG_INFO("Model file size: %d bytes", fileSize);

    while (fileReadIndex < fileSize)
    {
        bytesRead = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + fileReadIndex), EACH_READ_SIZE);
        if (bytesRead < 0)
            break;
        fileReadIndex += bytesRead;
    }
    
    ModelFileReader_Finish();
    
    if (fileReadIndex < fileSize)
    {
        LOG_ERROR("Incomplete model file read! Only read %d of %d bytes.", fileReadIndex, fileSize);
        return -2;
    }
    
    LOG_INFO("Model successfully loaded to HyperRAM.");
    return fileSize;
}

/* --- ML Inference and Post-Processing Task --- */
static void vInferenceTask(void *pvParameters)
{
    (void)pvParameters;
    
    // Load model from SD card
    int32_t modelSize = LoadModelFromSDCard();
    if (modelSize <= 0) {
        LOG_ERROR("Failed to load model from SD Card.");
        vTaskDelete(NULL);
        return;
    }

    // Initialize model wrapper
    arm::app::InferenceModel model;
    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    modelSize))
    {
        LOG_ERROR("Failed to initialize TFLite Micro model.");
        vTaskDelete(NULL);
        return;
    }

    // Initialize C++ post-processor
    arm::app::model::PostProcessor postProcessor(
        MODEL_INPUT_WIDTH, 
        MODEL_INPUT_HEIGHT, 
        MODEL_OUTPUT_GRID_SIZE, 
        MODEL_OUTPUT_GRID_SIZE
    );

    // Retrieve input & output tensors
    TfLiteTensor *inputTensor = model.GetInputTensor(0);
    TfLiteTensor *outputTensor = model.GetOutputTensor(0);

    /* Get model quantization parameters */
    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);
    arm::app::QuantParams outQuantParams = arm::app::GetTensorQuantParams(outputTensor);

    std::vector<arm::app::model::Detection> detections;
    
    // OpenMV Image structs for display
    image_t srcImg;
    srcImg.w = IMAGE_WIDTH;
    srcImg.h = IMAGE_HEIGHT;
    srcImg.size = FRAME_BUFFER_SIZE;
    srcImg.pixfmt = (IMAGE_CHANNELS == 3) ? PIXFORMAT_RGB888 : PIXFORMAT_GRAYSCALE;
    
    image_t dstImg;
    dstImg.w = 320;
    dstImg.h = 240;
    dstImg.size = IMAGE_FB_SIZE;
    dstImg.pixfmt = PIXFORMAT_RGB565;
    dstImg.data = (uint8_t *)frame_buf1;

    rectangle_t roi;
    roi.x = 0; roi.y = 0; roi.w = IMAGE_WIDTH; roi.h = IMAGE_HEIGHT;

#if defined(__EBI_LCD_PANEL__)
    S_DISP_RECT sDispRect;
    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

    uint64_t frameCount = 0;
    uint64_t lastTime = pmu_get_systick_Count();
    uint64_t currentFPS = 0;

    LOG_INFO("Inference Engine started. Waiting for incoming network video feed...");

    while (1)
    {
        // Wait for UDP receiver to signal a complete frame
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // 1. Quantize the raw grayscale pixels into the model input tensor (int8)
        int8_t *signedInputData = inputTensor->data.int8;
        
        for (int i = 0; i < FRAME_BUFFER_SIZE; ++i)
        {
            float pixelFloat = static_cast<float>(g_inferenceFrameBuffer[i]);
            
            // Adapt to trained normalization: [0, 1] vs raw [0, 255]
            float normalized = (inQuantParams.scale < 0.05f) ? (pixelFloat / 255.0f) : pixelFloat;
            int32_t quantized = static_cast<int32_t>(roundf(normalized / inQuantParams.scale)) + inQuantParams.offset;
            
            // Clip to int8 boundaries
            if (quantized < -128) quantized = -128;
            if (quantized > 127)  quantized = 127;
            
            signedInputData[i] = static_cast<int8_t>(quantized);
        }

        // 2. Execute Ethos-U Accelerated Inference
        model.RunInference();

        // 3. Post-Process Grid heatmaps to get target peaks (person locations)
        const int8_t *outputData = outputTensor->data.int8;
        postProcessor.Process(
            outputData,
            MODEL_DEFAULT_THRESHOLD,
            MODEL_MIN_PEAK_DISTANCE,
            outQuantParams.scale,
            outQuantParams.offset,
            detections
        );

        // 4. Update Metrics (FPS and Person Counts)
        frameCount++;
        uint64_t now = pmu_get_systick_Count();
        if (now - lastTime >= SystemCoreClock) // 1 second interval
        {
            currentFPS = frameCount;
            frameCount = 0;
            lastTime = now;
            
            LOG_INFO("[STATUS] Real-time inference rate: %llu FPS | Active People: %d", currentFPS, (int)detections.size());
        }

        // 5. Visual Rendering on LCD Panel (if enabled)
#if defined(__EBI_LCD_PANEL__)
        // Decompress/Upscale raw grayscale buffer to RGB565 LCD size
        srcImg.data = g_inferenceFrameBuffer;
        imlib_nvt_scale(&srcImg, &dstImg, &roi);

        // Draw crosshair indicators at each person peak
        for (const auto& det : detections)
        {
            // Map 192x192 model coordinates to 320x240 LCD coordinates
            int x_disp = static_cast<int>(det.x * (320.0f / 192.0f));
            int y_disp = static_cast<int>(det.y * (240.0f / 192.0f));

            // Draw a bounding crosshair around the detected center peak
            imlib_draw_rectangle(&dstImg, x_disp - 8, y_disp - 8, 16, 16, COLOR_R5_G6_B5_TO_RGB565(31, 0, 0), 2, false);
            
            // Draw center dot
            imlib_draw_rectangle(&dstImg, x_disp - 1, y_disp - 1, 2, 2, COLOR_R5_G6_B5_TO_RGB565(31, 31, 0), 1, true);
        }

        // Draw text overlay (FPS & count)
        char overlayText[64];
        sprintf(overlayText, "FPS: %llu | Count: %d", currentFPS, (int)detections.size());
        imlib_draw_string(&dstImg, 10, 10, overlayText, COLOR_R5_G6_B5_TO_RGB565(31, 31, 31), 2, 0, 0, false, false, false, false, 0, false, false);

        // Blit to screen
        sDispRect.u32TopLeftX = 0;
        sDispRect.u32TopLeftY = 0;
        sDispRect.u32BottonRightX = 319;
        sDispRect.u32BottonRightY = 239;
        Display_FillRect((uint16_t *)dstImg.data, &sDispRect, 1);
#endif
    }
}

/* LwIP TCP/IP stack thread task */
static void vNetworkInitTask(void *pvParameters)
{
    (void)pvParameters;
    ip_addr_t ipaddr, netmask, gw;

#if LWIP_DHCP_ENABLE
    IP4_ADDR(&gw, 0, 0, 0, 0);
    IP4_ADDR(&ipaddr, 0, 0, 0, 0);
    IP4_ADDR(&netmask, 0, 0, 0, 0);
#else
    ipaddr_aton(STATIC_IP_ADDR, &ipaddr);
    ipaddr_aton(STATIC_NETMASK, &netmask);
    ipaddr_aton(STATIC_GATEWAY, &gw);
#endif

    // Initialize TCP/IP core stack
    tcpip_init(NULL, NULL);

    // Register our Ethernet MAC (EMAC0) driver into LwIP netif
    netif_add(&g_netif, &ipaddr, &netmask, &gw, NULL, ethernetif_init, tcpip_input);
    netif_set_default(&g_netif);
    netif_set_up(&g_netif);

#if LWIP_DHCP_ENABLE
    LOG_INFO("DHCP starting...");
    if (dhcp_start(&g_netif) == ERR_OK)
    {
        while (dhcp_supplied_address(&g_netif) == 0)
        {
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    }
    else
    {
        LOG_ERROR("DHCP starting failed.");
        while (1);
    }
#endif

    LOG_INFO("Network stack successfully initialized.");
    LOG_INFO("IP address:      %s", ip4addr_ntoa(&g_netif.ip_addr));
    LOG_INFO("Subnet mask:     %s", ip4addr_ntoa(&g_netif.netmask));
    LOG_INFO("Default gateway: %s", ip4addr_ntoa(&g_netif.gw));

    // Spawn the high performance UDP Video Receiver thread
    xTaskCreate(vUdpVideoReceiverTask, "UdpRecv", 1024, NULL, tskIDLE_PRIORITY + 3UL, &xUdpReceiverTaskHandle);

    // Suspend network init task as it is no longer required
    vTaskSuspend(NULL);
}

int main(void)
{
    // Configure MPU regions for cache coherency and data access safety
    const ARM_MPU_Region_t mpuConfig[] =
    {
        {
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Only: 0=Read-Write, 1=Read-Only
                         0,                 // Non-Privileged: 0=Privileged & Non-Privileged, 1=Privileged only
                         1),                // eXecute Never: 0=Execution allowed, 1=Execution never allowed
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena) + sizeof(arm::app::tensorArena) - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - cacheable Write-Through
        },
        {
            ARM_MPU_RBAR(((unsigned int)fb_array),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Only
                         0,                 // Non-Privileged
                         1),                // eXecute Never
            ARM_MPU_RLAR((((unsigned int)fb_array) + sizeof(fb_array) - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // Non-Cacheable
        },
        {
            ARM_MPU_RBAR(((unsigned int)frame_buf1),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Only
                         0,                 // Non-Privileged
                         1),                // eXecute Never
            ARM_MPU_RLAR((((unsigned int)frame_buf1) + sizeof(frame_buf1) - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // Non-Cacheable
        }
    };

    // Apply custom MPU regions
    InitPreDefMPURegion(&mpuConfig[0], sizeof(mpuConfig) / sizeof(mpuConfig[0]));

    // Initialize target board (clocks, NPU, HyperRAM, SD card, Ethernet RMII pins)
    if (0 != BoardInit())
    {
        while (1);
    }

    // Initialize OpenMV memory allocators
    omv_init();

    LOG_INFO("----------------------------------------------------------------");
    LOG_INFO("    Starting M55M1 UDP Server People Counting Firmware     ");
    LOG_INFO("----------------------------------------------------------------");

    // Create system coordinator tasks
    xTaskCreate(vNetworkInitTask, "NetInit", TCPIP_THREAD_STACKSIZE, NULL, tskIDLE_PRIORITY + 4UL, NULL);
    xTaskCreate(vInferenceTask, "Inference", 2048, NULL, tskIDLE_PRIORITY + 2UL, &xInferenceTaskHandle);

    // Start FreeRTOS scheduler
    vTaskStartScheduler();

    // System will never reach here unless memory allocation failed
    for (;;);
}

/* --- FREERTOS HOOKS --- */
extern "C" {
void vApplicationMallocFailedHook(void)
{
    LOG_ERROR("FreeRTOS Stack Overflow / Memory allocation failed!");
    taskDISABLE_INTERRUPTS();
    for (;;);
}

void vApplicationIdleHook(void) {}

void vApplicationStackOverflowHook(TaskHandle_t pxTask, char *pcTaskName)
{
    (void)pcTaskName;
    (void)pxTask;
    LOG_ERROR("FreeRTOS Stack Overflow detected in task: %s", pcTaskName);
    taskDISABLE_INTERRUPTS();
    for (;;);
}

void vApplicationTickHook(void) {}
}
