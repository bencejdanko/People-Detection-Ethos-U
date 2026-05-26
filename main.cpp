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
#include <cmath>
#include <cstring>

/* FreeRTOS includes */
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

/* LwIP network includes */
extern "C" {
#include "lwip/tcpip.h"
#include "netif/ethernetif.h"
#include "lwip/udp.h"
#include "lwip/pbuf.h"
#include "lwip/def.h"
#include "emac.h"
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
#include "embedded_model.h"

#if defined(__EBI_LCD_PANEL__)
#include "Display.h"
#endif

/* Task Handles */
static TaskHandle_t xInferenceTaskHandle = NULL;

/* Network interface structure */
struct netif g_netif;
static char g_deviceIpAddress[IPADDR_STRLEN_MAX] = "0.0.0.0";

/* --- FRAME BUFFER MEMORY ALLOCATION --- */
__attribute__((section(".bss.hyperram.data"), aligned(32))) static uint8_t g_udpFrameBuffers[3][FRAME_BUFFER_SIZE];
static volatile int32_t g_rxFrameBufferIndex = 0;
static volatile int32_t g_publishedFrameBufferIndex = -1;
static volatile int32_t g_processingFrameBufferIndex = -1;

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
__attribute__((section(".bss.hyperram.data"), aligned(32))) static char frame_buf1[LCD_FRAME_BUFFER_SIZE];

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static inline uint16_t Rgb888ToRgb565(const uint8_t *pixel)
{
    return (uint16_t)(((uint16_t)(pixel[0] & 0xF8) << 8) |
                      ((uint16_t)(pixel[1] & 0xFC) << 3) |
                      ((uint16_t)pixel[2] >> 3));
}

static uint16_t s_displayXMap[LCD_DISPLAY_WIDTH];
static uint16_t s_displayYMap[LCD_DISPLAY_HEIGHT];

#if defined(__EBI_LCD_PANEL__)
static const int kStatusTextScale = 4;
static const int kStatusTextMargin = 16;
static const int kStatusTextLineHeight = FONT_HTIGHT * kStatusTextScale + 8;
static const int kStatusTextColor = COLOR_R5_G6_B5_TO_RGB565(31, 63, 31);
static const int kStatusBackgroundColor = COLOR_R5_G6_B5_TO_RGB565(0, 0, 0);

static void CopyDeviceIpAddress(char *dst, size_t dstSize)
{
    if (dstSize == 0)
    {
        return;
    }

    if (xTaskGetSchedulerState() != taskSCHEDULER_NOT_STARTED)
    {
        taskENTER_CRITICAL();
    }
    strncpy(dst, g_deviceIpAddress, dstSize - 1);
    dst[dstSize - 1] = '\0';
    if (xTaskGetSchedulerState() != taskSCHEDULER_NOT_STARTED)
    {
        taskEXIT_CRITICAL();
    }
}

static int StatusTextX(const char *text)
{
    const int textWidth = (int)strlen(text) * FONT_WIDTH * kStatusTextScale;
    const int x = LCD_DISPLAY_WIDTH - kStatusTextMargin - textWidth;

    return (x > kStatusTextMargin) ? x : kStatusTextMargin;
}

static void DrawStatusOverlay(image_t *img, uint64_t fps, size_t peopleCount)
{
    char ipAddress[IPADDR_STRLEN_MAX];
    char lines[3][40];

    CopyDeviceIpAddress(ipAddress, sizeof(ipAddress));

    sprintf(lines[0], "FPS: %llu", fps);
    sprintf(lines[1], "PEOPLE: %d", (int)peopleCount);
    sprintf(lines[2], "IP: %s", ipAddress);

    int overlayWidth = 0;
    for (size_t i = 0; i < 3; ++i)
    {
        const int lineWidth = (int)strlen(lines[i]) * FONT_WIDTH * kStatusTextScale;
        if (lineWidth > overlayWidth)
        {
            overlayWidth = lineWidth;
        }
    }
    overlayWidth += kStatusTextMargin;

    int overlayX = LCD_DISPLAY_WIDTH - kStatusTextMargin - overlayWidth;
    if (overlayX < kStatusTextMargin)
    {
        overlayX = kStatusTextMargin;
        overlayWidth = LCD_DISPLAY_WIDTH - (kStatusTextMargin * 2);
    }
    const int overlayHeight = (kStatusTextLineHeight * 3) + kStatusTextMargin;
    imlib_draw_rectangle(img,
                         overlayX,
                         kStatusTextMargin,
                         overlayWidth,
                         overlayHeight,
                         kStatusBackgroundColor,
                         1,
                         true);

    imlib_draw_string(img, StatusTextX(lines[0]), kStatusTextMargin, lines[0], kStatusTextColor, kStatusTextScale, 0, 0, false, false, false, false, 0, false, false);
    imlib_draw_string(img, StatusTextX(lines[1]), kStatusTextMargin + kStatusTextLineHeight, lines[1], kStatusTextColor, kStatusTextScale, 0, 0, false, false, false, false, 0, false, false);
    imlib_draw_string(img, StatusTextX(lines[2]), kStatusTextMargin + (kStatusTextLineHeight * 2), lines[2], kStatusTextColor, kStatusTextScale, 0, 0, false, false, false, false, 0, false, false);
}

static void DrawStatusScreen(uint64_t fps, size_t peopleCount)
{
    char ipAddress[IPADDR_STRLEN_MAX];
    char line[40];

    CopyDeviceIpAddress(ipAddress, sizeof(ipAddress));
    Display_ClearLCD(C_BLACK);

    sprintf(line, "FPS: %llu", fps);
    Display_PutText(line, strlen(line), StatusTextX(line), kStatusTextMargin, C_WHITE, C_BLACK, false, kStatusTextScale);

    sprintf(line, "PEOPLE: %d", (int)peopleCount);
    Display_PutText(line, strlen(line), StatusTextX(line), kStatusTextMargin + kStatusTextLineHeight, C_WHITE, C_BLACK, false, kStatusTextScale);

    sprintf(line, "IP: %s", ipAddress);
    Display_PutText(line, strlen(line), StatusTextX(line), kStatusTextMargin + (kStatusTextLineHeight * 2), C_WHITE, C_BLACK, false, kStatusTextScale);
}
#endif

static void ConvertRgb888ToRgb565Scaled(const uint8_t *src, uint16_t *dst, uint32_t dstWidth, uint32_t dstHeight)
{
    static bool mapsInitialized = false;

    if (!mapsInitialized)
    {
        for (uint32_t x = 0; x < dstWidth; ++x)
        {
            s_displayXMap[x] = (uint16_t)((x * IMAGE_WIDTH) / dstWidth);
        }

        for (uint32_t y = 0; y < dstHeight; ++y)
        {
            s_displayYMap[y] = (uint16_t)((y * IMAGE_HEIGHT) / dstHeight);
        }

        mapsInitialized = true;
    }

    for (uint32_t y = 0; y < dstHeight; ++y)
    {
        const uint8_t *srcRow = src + (s_displayYMap[y] * IMAGE_WIDTH * IMAGE_CHANNELS);
        uint16_t *dstRow = dst + (y * dstWidth);

        for (uint32_t x = 0; x < dstWidth; ++x)
        {
            dstRow[x] = Rgb888ToRgb565(srcRow + (s_displayXMap[x] * IMAGE_CHANNELS));
        }
    }
}

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
    uint32_t total_len;     // Expected total size of raw pixels
    uint32_t chunk_offset;  // Offset of this chunk in the frame buffer
    uint32_t chunk_len;     // Length of this chunk payload
};

static struct udp_pcb *s_udpVideoPcb = NULL;
static uint32_t s_udpActiveFrameId = 0xFFFFFFFF;
static uint32_t s_udpAccumulatedBytes = 0;
static bool s_udpActiveFramePublished = false;

static int32_t SelectNextRxFrameBuffer(void)
{
    for (int32_t i = 0; i < 3; ++i)
    {
        if ((i != g_publishedFrameBufferIndex) && (i != g_processingFrameBufferIndex))
        {
            return i;
        }
    }

    return g_rxFrameBufferIndex;
}

static void PublishReceivedFrame(void)
{
    taskENTER_CRITICAL();
    g_publishedFrameBufferIndex = g_rxFrameBufferIndex;
    s_udpActiveFramePublished = true;
    g_rxFrameBufferIndex = SelectNextRxFrameBuffer();
    taskEXIT_CRITICAL();

    if (xInferenceTaskHandle != NULL)
    {
        xTaskNotifyGive(xInferenceTaskHandle);
    }
}

static void UdpVideoReceiveCallback(void *arg,
                                    struct udp_pcb *pcb,
                                    struct pbuf *p,
                                    const ip_addr_t *addr,
                                    u16_t port)
{
    (void)arg;
    (void)pcb;
    (void)addr;
    (void)port;

    if (p == NULL)
    {
        return;
    }

    if (p->tot_len >= sizeof(PacketHeader))
    {
        PacketHeader header;
        if (pbuf_copy_partial(p, &header, sizeof(header), 0) == sizeof(header))
        {
            uint32_t magic = ntohl(header.magic);
            uint32_t frameId = ntohl(header.frame_id);
            uint32_t totalLen = ntohl(header.total_len);
            uint32_t chunkOffset = ntohl(header.chunk_offset);
            uint32_t chunkLen = ntohl(header.chunk_len);

            if ((magic == UDP_MAGIC_HEADER) &&
                (totalLen == FRAME_BUFFER_SIZE) &&
                ((chunkOffset + chunkLen) <= FRAME_BUFFER_SIZE) &&
                (p->tot_len >= (sizeof(PacketHeader) + chunkLen)))
            {
                if (chunkOffset == 0)
                {
                    s_udpActiveFrameId = frameId;
                    s_udpAccumulatedBytes = 0;
                    s_udpActiveFramePublished = false;
                }

                if ((frameId == s_udpActiveFrameId) && !s_udpActiveFramePublished)
                {
                    uint8_t *rxFrame = g_udpFrameBuffers[g_rxFrameBufferIndex];
                    if (pbuf_copy_partial(p,
                                          rxFrame + chunkOffset,
                                          (u16_t)chunkLen,
                                          sizeof(PacketHeader)) == chunkLen)
                    {
                        s_udpAccumulatedBytes += chunkLen;

                        if (s_udpAccumulatedBytes == FRAME_BUFFER_SIZE)
                        {
                            PublishReceivedFrame();
                        }
                    }
                }
            }
        }
    }

    pbuf_free(p);
}

static void UdpVideoInitCallback(void *arg)
{
    (void)arg;

    s_udpVideoPcb = udp_new();
    if (s_udpVideoPcb == NULL)
    {
        LOG_ERROR("Failed to create raw UDP PCB.");
        return;
    }

    err_t err = udp_bind(s_udpVideoPcb, IP_ADDR_ANY, UDP_STREAM_PORT);
    if (err != ERR_OK)
    {
        LOG_ERROR("Failed to bind raw UDP PCB to port %d: lwIP err=%d", UDP_STREAM_PORT, (int)err);
        udp_remove(s_udpVideoPcb);
        s_udpVideoPcb = NULL;
        return;
    }

    udp_recv(s_udpVideoPcb, UdpVideoReceiveCallback, NULL);
    LOG_INFO("Raw UDP video receiver listening on port %d.", UDP_STREAM_PORT);
}

static bool EmbeddedModelContainsEthosUCustomOp(void)
{
    static const char ethosUOpName[] = "ethos-u";
    const size_t opNameLen = sizeof(ethosUOpName) - 1;

    if (g_model_tflite_len < opNameLen) {
        return false;
    }

    for (unsigned int i = 0; i <= g_model_tflite_len - opNameLen; ++i) {
        if (std::memcmp(&g_model_tflite[i], ethosUOpName, opNameLen) == 0) {
            return true;
        }
    }

    return false;
}

/* Use the baked-in model from internal flash. Ethos-U can read this region. */
static int32_t LoadModelFromEmbeddedFlash(const unsigned char **modelData)
{
    *modelData = g_model_tflite;
    LOG_INFO("Using model directly from embedded flash (%u bytes).", g_model_tflite_len);
    return (int32_t)g_model_tflite_len;
}

/* --- ML Inference and Post-Processing Task --- */
static void vInferenceTask(void *pvParameters)
{
    (void)pvParameters;
    
    // Use model from embedded flash.
    const unsigned char *modelData = NULL;
    int32_t modelSize = LoadModelFromEmbeddedFlash(&modelData);
    if (modelSize <= 0) {
        LOG_ERROR("Failed to load embedded model.");
        vTaskDelete(NULL);
        return;
    }

    if (!EmbeddedModelContainsEthosUCustomOp()) {
        LOG_ERROR("Embedded model is not Vela-optimized for Ethos-U (missing custom op \"ethos-u\").");
        LOG_ERROR("Regenerate embedded_model.h from vela_output/*.tflite, not the original int8 model.");
        vTaskDelete(NULL);
        return;
    }

    // Initialize model wrapper
    arm::app::InferenceModel model;
    LOG_INFO("TFLM tensor arena: start=0x%08X end=0x%08X size=%u bytes.",
             (unsigned int)arm::app::tensorArena,
             (unsigned int)(arm::app::tensorArena + sizeof(arm::app::tensorArena) - 1),
             (unsigned int)sizeof(arm::app::tensorArena));
    LOG_INFO("TFLM model buffer: start=0x%08X end=0x%08X size=%d bytes.",
             (unsigned int)modelData,
             (unsigned int)(modelData + modelSize - 1),
             (int)modelSize);
    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    modelData,
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

    arm::app::model::Detection detections[MODEL_MAX_DETECTIONS] = {};
    size_t detectionCount = 0;
    
    const bool useFastInputQuant =
        (IMAGE_CHANNELS == 3) &&
        (inQuantParams.offset == -1) &&
        (inQuantParams.scale > 0.007f) &&
        (inQuantParams.scale < 0.0085f);

    // OpenMV image struct for overlays/text on the full LCD display buffer.
    image_t dstImg;
    dstImg.w = LCD_DISPLAY_WIDTH;
    dstImg.h = LCD_DISPLAY_HEIGHT;
    dstImg.size = LCD_FRAME_BUFFER_SIZE;
    dstImg.pixfmt = PIXFORMAT_RGB565;
    dstImg.data = (uint8_t *)frame_buf1;

#if defined(__EBI_LCD_PANEL__)
    S_DISP_RECT sDispRect;
#endif

    uint64_t frameCount = 0;
    uint64_t lastTime = pmu_get_systick_Count();
    uint64_t currentFPS = 0;

    LOG_INFO("Inference Engine initialized. Stack high water mark: %u words remaining.", (unsigned int)uxTaskGetStackHighWaterMark(NULL));
    LOG_INFO("Waiting for incoming network video feed...");

    while (1)
    {
        // Wait for the raw UDP callback to publish a complete frame.
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        taskENTER_CRITICAL();
        int32_t frameBufferIndex = g_publishedFrameBufferIndex;
        g_processingFrameBufferIndex = frameBufferIndex;
        taskEXIT_CRITICAL();

        if (frameBufferIndex < 0)
        {
            continue;
        }

        const uint8_t *inferenceFrame = g_udpFrameBuffers[frameBufferIndex];

        // 1. Quantize the raw RGB pixels into the model input tensor (int8)
        int8_t *signedInputData = inputTensor->data.int8;

        if (useFastInputQuant)
        {
            for (int i = 0; i < FRAME_BUFFER_SIZE; ++i)
            {
                signedInputData[i] = static_cast<int8_t>((((uint16_t)inferenceFrame[i] + 1U) >> 1) - 1);
            }
        }
        else
        {
            for (int i = 0; i < FRAME_BUFFER_SIZE; ++i)
            {
                float pixelFloat = static_cast<float>(inferenceFrame[i]);
                float normalized = (inQuantParams.scale < 0.05f) ? (pixelFloat / 255.0f) : pixelFloat;
                int32_t quantized = static_cast<int32_t>(roundf(normalized / inQuantParams.scale)) + inQuantParams.offset;

                if (quantized < -128) quantized = -128;
                if (quantized > 127)  quantized = 127;

                signedInputData[i] = static_cast<int8_t>(quantized);
            }
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
            detections,
            MODEL_MAX_DETECTIONS,
            detectionCount
        );

        // 4. Update Metrics (FPS and Person Counts)
        frameCount++;
        uint64_t now = pmu_get_systick_Count();
        if (now - lastTime >= SystemCoreClock) // 1 second interval
        {
            currentFPS = frameCount;
            frameCount = 0;
            lastTime = now;
            
            LOG_INFO("[STATUS] Real-time inference rate: %llu FPS | Active People: %d", currentFPS, (int)detectionCount);
        }

        // 5. Visual Rendering on LCD Panel (if enabled)
#if defined(__EBI_LCD_PANEL__)
        // Scale the raw RGB888 input frame to fill the LCD and convert to RGB565.
        ConvertRgb888ToRgb565Scaled(inferenceFrame,
                                    (uint16_t *)dstImg.data,
                                    LCD_DISPLAY_WIDTH,
                                    LCD_DISPLAY_HEIGHT);

        // Draw crosshair indicators at each person peak
        for (size_t i = 0; i < detectionCount; ++i)
        {
            const arm::app::model::Detection& det = detections[i];
            int x_disp = static_cast<int>((det.x * LCD_DISPLAY_WIDTH) / IMAGE_WIDTH);
            int y_disp = static_cast<int>((det.y * LCD_DISPLAY_HEIGHT) / IMAGE_HEIGHT);
            int box_w = LCD_DISPLAY_WIDTH / 24;
            int box_h = LCD_DISPLAY_HEIGHT / 24;
            if (box_w < 8) box_w = 8;
            if (box_h < 8) box_h = 8;

            // Draw a bounding crosshair around the detected center peak
            imlib_draw_rectangle(&dstImg,
                                 x_disp - (box_w / 2),
                                 y_disp - (box_h / 2),
                                 box_w,
                                 box_h,
                                 COLOR_R5_G6_B5_TO_RGB565(31, 0, 0),
                                 2,
                                 false);
            
            // Draw center dot
            imlib_draw_rectangle(&dstImg, x_disp - 1, y_disp - 1, 2, 2, COLOR_R5_G6_B5_TO_RGB565(31, 31, 0), 1, true);
        }

        DrawStatusOverlay(&dstImg, currentFPS, detectionCount);

        // Blit to screen
        sDispRect.u32TopLeftX = 0;
        sDispRect.u32TopLeftY = 0;
        sDispRect.u32BottonRightX = LCD_DISPLAY_WIDTH - 1;
        sDispRect.u32BottonRightY = LCD_DISPLAY_HEIGHT - 1;
        Display_FillRect((uint16_t *)dstImg.data, &sDispRect, 1);
#endif
        taskENTER_CRITICAL();
        if (g_processingFrameBufferIndex == frameBufferIndex)
        {
            g_processingFrameBufferIndex = -1;
        }
        taskEXIT_CRITICAL();
    }
}

/* LwIP TCP/IP stack thread task */
static void vNetworkInitTask(void *pvParameters)
{
    (void)pvParameters;
    ip_addr_t ipaddr, netmask, gw;
    struct netif *netifResult;

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
    LOG_INFO("Initializing LwIP TCP/IP core stack...");
    LOG_INFO("FreeRTOS heap before tcpip_init: %u bytes.", (unsigned int)xPortGetFreeHeapSize());
    tcpip_init(NULL, NULL);
    LOG_INFO("LwIP TCP/IP core stack initialized successfully.");
    LOG_INFO("FreeRTOS heap after tcpip_init: %u bytes.", (unsigned int)xPortGetFreeHeapSize());

    // Register our Ethernet MAC (EMAC0) driver into LwIP netif
    LOG_INFO("Registering Ethernet MAC driver (EMAC0) to LwIP...");
    netifResult = netif_add(&g_netif, &ipaddr, &netmask, &gw, NULL, ethernetif_init, tcpip_input);
    if (netifResult == NULL)
    {
        LOG_ERROR("netif_add failed while registering EMAC0. FreeRTOS heap remaining: %u bytes.",
                  (unsigned int)xPortGetFreeHeapSize());
        vTaskDelete(NULL);
        return;
    }
    LOG_INFO("netif_add returned successfully. FreeRTOS heap remaining: %u bytes.",
             (unsigned int)xPortGetFreeHeapSize());
    LOG_INFO("Setting EMAC0 as the default LwIP interface...");
    netif_set_default(&g_netif);
    LOG_INFO("Bringing EMAC0 interface UP...");
    netif_set_up(&g_netif);
    LOG_INFO("Ethernet interface registered and brought UP successfully.");

#if LWIP_DHCP_ENABLE
    LOG_INFO("Requesting IP address via DHCP...");
    if (dhcp_start(&g_netif) == ERR_OK)
    {
        while (dhcp_supplied_address(&g_netif) == 0)
        {
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    }
    else
    {
        LOG_ERROR("DHCP starting failed!");
        while (1);
    }
#endif

    LOG_INFO("Network interface successfully configured:");
    LOG_INFO("  IP address:      %s", ip4addr_ntoa(&g_netif.ip_addr));
    LOG_INFO("  Subnet mask:     %s", ip4addr_ntoa(&g_netif.netmask));
    LOG_INFO("  Default gateway: %s", ip4addr_ntoa(&g_netif.gw));

#if defined(__EBI_LCD_PANEL__)
    taskENTER_CRITICAL();
    ipaddr_ntoa_r(&g_netif.ip_addr, g_deviceIpAddress, sizeof(g_deviceIpAddress));
    taskEXIT_CRITICAL();
    DrawStatusScreen(0, 0);
#endif

    LOG_INFO("Registering raw UDP video receiver callback...");
    if (tcpip_callback(UdpVideoInitCallback, NULL) != ERR_OK)
    {
        LOG_ERROR("Failed to schedule raw UDP receiver initialization.");
        vTaskDelete(NULL);
        return;
    }

    LOG_INFO("Network initialization complete.");
    LOG_INFO("NetInit stack high water mark: %u words remaining.", (unsigned int)uxTaskGetStackHighWaterMark(NULL));
    LOG_INFO("Suspending NetInit task.");
    vTaskSuspend(NULL);
}

int main(void)
{
    // Initialize target board (clocks, NPU, HyperRAM, SD card, Ethernet RMII pins)
    if (0 != BoardInit())
    {
        while (1);
    }

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
    LOG_INFO("Configuring custom MPU memory caching regions...");
    InitPreDefMPURegion(&mpuConfig[0], sizeof(mpuConfig) / sizeof(mpuConfig[0]));
    LOG_INFO("Custom MPU memory cache regions applied successfully (tensorArena WTRA, framebuffers non-cacheable).");

    // Enable MemManage, BusFault, and UsageFault handlers explicitly in SCB
    LOG_INFO("Enabling dedicated SCB fault handlers (MemManage, BusFault, UsageFault)...");
    SCB->SHCSR |= (SCB_SHCSR_MEMFAULTENA_Msk | SCB_SHCSR_BUSFAULTENA_Msk | SCB_SHCSR_USGFAULTENA_Msk);
    LOG_INFO("SCB fault handlers activated successfully.");

    // Initialize OpenMV memory allocators
    LOG_INFO("Initializing OpenMV frame buffer allocators...");
    omv_init();
    LOG_INFO("OpenMV frame buffer memory allocator initialized successfully.");

#if defined(__EBI_LCD_PANEL__)
    LOG_INFO("Initializing LCD panel...");
    Display_Init();
    LOG_INFO("LCD panel initialization returned. Drawing status screen...");
    DrawStatusScreen(0, 0);
    LOG_INFO("LCD status screen drawn.");
#endif

    LOG_INFO("----------------------------------------------------------------");
    LOG_INFO("    Starting M55M1 UDP Server People Counting Firmware     ");
    LOG_INFO("----------------------------------------------------------------");

    // Create system coordinator tasks
    LOG_INFO("Spawning NetworkInit FreeRTOS task (Stack: 2048 words, Priority: %lu)...", (unsigned long)(tskIDLE_PRIORITY + 4UL));
    if (xTaskCreate(vNetworkInitTask, "NetInit", 2048, NULL, tskIDLE_PRIORITY + 4UL, NULL) != pdPASS)
    {
        LOG_ERROR("Failed to create NetworkInit task. FreeRTOS heap remaining: %u bytes.",
                  (unsigned int)xPortGetFreeHeapSize());
        while (1);
    }
    
    LOG_INFO("Spawning ML Inference FreeRTOS task (Stack: 4096 words, Priority: %lu)...", (unsigned long)(tskIDLE_PRIORITY + 2UL));
    if (xTaskCreate(vInferenceTask, "Inference", 4096, NULL, tskIDLE_PRIORITY + 2UL, &xInferenceTaskHandle) != pdPASS)
    {
        LOG_ERROR("Failed to create ML Inference task. FreeRTOS heap remaining: %u bytes.",
                  (unsigned int)xPortGetFreeHeapSize());
        while (1);
    }

    // Start FreeRTOS scheduler
    LOG_INFO("Starting FreeRTOS Task Scheduler...");
    vTaskStartScheduler();

    // System will never reach here unless memory allocation failed
    for (;;);
}

/* --- FREERTOS HOOKS & DIAGNOSTIC FAULT HANDLERS --- */
extern "C" {

// Crash-safe fault analyzer - reads SCB registers FIRST (no stack dependency)
void HardFault_Handler_C(uint32_t *pulStackedRegisters)
{
    // Read SCB fault registers immediately (memory-mapped, always accessible)
    volatile uint32_t cfsr  = SCB->CFSR;
    volatile uint32_t hfsr  = SCB->HFSR;
    volatile uint32_t mmfar = SCB->MMFAR;
    volatile uint32_t bfar  = SCB->BFAR;

    printf("\r\n==================================================\r\n");
    printf("   HARDWARE FAULT DETECTED\r\n");
    printf("==================================================\r\n");

    // Decode CFSR fault type
    printf("CFSR  = 0x%08X\r\n", (unsigned int)cfsr);
    if (cfsr & 0xFF) {
        printf("  >> MemManage Fault:\r\n");
        if (cfsr & (1 << 0)) printf("     IACCVIOL  - Instruction access violation\r\n");
        if (cfsr & (1 << 1)) printf("     DACCVIOL  - Data access violation\r\n");
        if (cfsr & (1 << 3)) printf("     MUNSTKERR - Unstacking error\r\n");
        if (cfsr & (1 << 4)) printf("     MSTKERR   - Stacking error (SP invalid)\r\n");
        if (cfsr & (1 << 5)) printf("     MLSPERR   - FP lazy state error\r\n");
        if (cfsr & (1 << 7)) printf("     MMARVALID - Faulting addr in MMFAR\r\n");
    }
    if (cfsr & 0xFF00) {
        printf("  >> BusFault:\r\n");
        if (cfsr & (1 << 8))  printf("     IBUSERR    - Instruction bus error\r\n");
        if (cfsr & (1 << 9))  printf("     PRECISERR  - Precise data bus error\r\n");
        if (cfsr & (1 << 10)) printf("     IMPRECISERR- Imprecise data bus error\r\n");
        if (cfsr & (1 << 11)) printf("     UNSTKERR   - Unstacking bus error\r\n");
        if (cfsr & (1 << 12)) printf("     STKERR     - Stacking bus error\r\n");
        if (cfsr & (1 << 13)) printf("     LSPERR     - FP lazy stacking error\r\n");
        if (cfsr & (1 << 15)) printf("     BFARVALID  - Faulting addr in BFAR\r\n");
    }
    if (cfsr & 0xFFFF0000) {
        printf("  >> UsageFault:\r\n");
        if (cfsr & (1 << 16)) printf("     UNDEFINSTR - Undefined instruction\r\n");
        if (cfsr & (1 << 17)) printf("     INVSTATE   - Invalid EPSR state\r\n");
        if (cfsr & (1 << 18)) printf("     INVPC      - Invalid EXC_RETURN\r\n");
        if (cfsr & (1 << 19)) printf("     NOCP       - Coprocessor not enabled\r\n");
        if (cfsr & (1 << 24)) printf("     UNALIGNED  - Unaligned access\r\n");
        if (cfsr & (1 << 25)) printf("     DIVBYZERO  - Divide by zero\r\n");
    }

    printf("HFSR  = 0x%08X\r\n", (unsigned int)hfsr);
    if (hfsr & (1 << 30)) printf("  >> FORCED - Escalated from configurable fault\r\n");
    if (hfsr & (1 << 1))  printf("  >> VECTTBL - Vector table read fault\r\n");
    printf("MMFAR = 0x%08X\r\n", (unsigned int)mmfar);
    printf("BFAR  = 0x%08X\r\n", (unsigned int)bfar);

    // Validate stack pointer before dereferencing
    uint32_t sp = (uint32_t)pulStackedRegisters;
    if ((sp >= 0x20000000) && (sp < 0x92000000) && ((sp & 0x3) == 0)) {
        printf("\r\nStacked Registers (SP=0x%08X):\r\n", sp);
        printf("  R0  = 0x%08X   R1  = 0x%08X\r\n", pulStackedRegisters[0], pulStackedRegisters[1]);
        printf("  R2  = 0x%08X   R3  = 0x%08X\r\n", pulStackedRegisters[2], pulStackedRegisters[3]);
        printf("  R12 = 0x%08X   LR  = 0x%08X\r\n", pulStackedRegisters[4], pulStackedRegisters[5]);
        printf("  PC  = 0x%08X   PSR = 0x%08X\r\n", pulStackedRegisters[6], pulStackedRegisters[7]);
    } else {
        printf("\r\n  !! SP CORRUPTED: 0x%08X (not in valid RAM)\r\n", sp);
        printf("  !! Cannot read stacked registers -- likely STACK OVERFLOW\r\n");
    }

    // Print active FreeRTOS task if scheduler is running
    if (xTaskGetSchedulerState() != taskSCHEDULER_NOT_STARTED) {
        TaskHandle_t current = xTaskGetCurrentTaskHandle();
        if (current != NULL) {
            printf("  Active task: \"%s\"\r\n", pcTaskGetName(current));
        }
    }

    printf("==================================================\r\n");
    // Only halt if a debugger is attached; otherwise just spin
    if (CoreDebug->DHCSR & CoreDebug_DHCSR_C_DEBUGEN_Msk) {
        __BKPT(0);
    }
    while (1);
}

// Assembly wrappers to retrieve the active stack pointer (MSP/PSP)
__attribute__((naked)) void HardFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4\n"
        "ite eq\n"
        "mrseq r0, msp\n"
        "mrsne r0, psp\n"
        "b HardFault_Handler_C\n"
    );
}

__attribute__((naked)) void MemManage_Handler(void)
{
    __asm volatile(
        "tst lr, #4\n"
        "ite eq\n"
        "mrseq r0, msp\n"
        "mrsne r0, psp\n"
        "b HardFault_Handler_C\n"
    );
}

__attribute__((naked)) void BusFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4\n"
        "ite eq\n"
        "mrseq r0, msp\n"
        "mrsne r0, psp\n"
        "b HardFault_Handler_C\n"
    );
}

__attribute__((naked)) void UsageFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4\n"
        "ite eq\n"
        "mrseq r0, msp\n"
        "mrsne r0, psp\n"
        "b HardFault_Handler_C\n"
    );
}

void vApplicationMallocFailedHook(void)
{
    LOG_ERROR("FreeRTOS Heap allocation failed!");
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

void vApplicationTickHook(void)
{
    FreeRTOS_TickHook((uint32_t)xTaskGetTickCountFromISR());
}
}
