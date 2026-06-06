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

/* LwIP network includes (UDP server path only) */
#if !USE_CCAP_CAMERA
extern "C" {
#include "lwip/tcpip.h"
#include "netif/ethernetif.h"
#include "lwip/dhcp.h"
#include "lwip/udp.h"
#include "lwip/pbuf.h"
#include "lwip/def.h"
#include "emac.h"
}
#endif /* !USE_CCAP_CAMERA */

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

/* CCAP onboard camera (when USE_CCAP_CAMERA is enabled) */
#if USE_CCAP_CAMERA
extern "C" {
#include "ImageSensor.h"
}
#endif /* USE_CCAP_CAMERA */

/* Model and ML includes */
#include "InferenceModel.hpp"
#include "PostProcessor.hpp"
#include "embedded_model.h"

#if defined(__EBI_LCD_PANEL__)
#include "Display.h"
#endif

/* Task Handles */
static TaskHandle_t xInferenceTaskHandle = NULL;

#if !USE_CCAP_CAMERA
/* Network interface structure (UDP path only) */
struct netif g_netif;
static char g_deviceIpAddress[IPADDR_STRLEN_MAX] = "0.0.0.0";

/* --- UDP FRAME BUFFER MEMORY ALLOCATION --- */
__attribute__((section(".bss.hyperram.data"), aligned(32))) static uint8_t g_udpFrameBuffers[3][FRAME_BUFFER_SIZE];
static volatile int32_t g_rxFrameBufferIndex = 0;
static volatile int32_t g_publishedFrameBufferIndex = -1;
static volatile int32_t g_processingFrameBufferIndex = -1;
#else
/* --- CCAP FRAME BUFFER MEMORY ALLOCATION ---
 * Two QVGA (320x240) RGB565 ping-pong buffers so that CCAP DMA into one
 * buffer can overlap with NPU inference on the other. */
#define CCAP_CAPTURE_WIDTH   320
#define CCAP_CAPTURE_HEIGHT  240
#define CCAP_FB_SIZE         (CCAP_CAPTURE_WIDTH * CCAP_CAPTURE_HEIGHT * 2)  // RGB565
__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t g_ccapBuf0[CCAP_FB_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t g_ccapBuf1[CCAP_FB_SIZE];
#endif /* !USE_CCAP_CAMERA */

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

__attribute__((section(".bss.hyperram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.hyperram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];
__attribute__((section(".bss.hyperram.data"), aligned(32))) static char frame_buf1[LCD_FRAME_BUFFER_SIZE];
__attribute__((section(".bss.hyperram.data"), aligned(32))) static char frame_buf2[LCD_FRAME_BUFFER_SIZE];

static char *g_lcdDrawBuf = frame_buf1;
static char *g_lcdShowBuf = frame_buf2;
static volatile bool g_lcdBlitPending = false;
static TaskHandle_t xDisplayTaskHandle = NULL;

#if USE_CCAP_CAMERA
    #define RENDER_WIDTH   CCAP_CAPTURE_WIDTH
    #define RENDER_HEIGHT  CCAP_CAPTURE_HEIGHT
#else
    #define RENDER_WIDTH   LCD_DISPLAY_WIDTH
    #define RENDER_HEIGHT  LCD_DISPLAY_HEIGHT
#endif
#define RENDER_FB_SIZE (RENDER_WIDTH * RENDER_HEIGHT * 2)

static int8_t g_quantLUT[256];
static volatile uint32_t g_displayBlitCycles = 0;

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

#if !USE_CCAP_CAMERA
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
#endif /* !USE_CCAP_CAMERA */

static void DrawStatusOverlay(image_t *img, uint64_t fps, size_t peopleCount)
{
    char lines[3][40];
    int textScale = (img->h >= 480) ? 4 : 2;
    int lineSpacing = FONT_HTIGHT * textScale + 8;
    const int statusX = kStatusTextMargin;
    const int statusY = img->h - kStatusTextMargin - (lineSpacing * 3);

    sprintf(lines[0], "FPS: %llu", fps);
    sprintf(lines[1], "PEOPLE: %d", (int)peopleCount);
#if USE_CCAP_CAMERA
    sprintf(lines[2], "SRC: CCAP Camera");
#else
    char ipAddress[IPADDR_STRLEN_MAX];
    CopyDeviceIpAddress(ipAddress, sizeof(ipAddress));
    sprintf(lines[2], "IP: %s", ipAddress);
#endif

    imlib_draw_string(img, statusX, statusY, lines[0], kStatusTextColor, textScale, 0, 0, false, false, false, false, 0, false, false);
    imlib_draw_string(img, statusX, statusY + lineSpacing, lines[1], kStatusTextColor, textScale, 0, 0, false, false, false, false, 0, false, false);
    imlib_draw_string(img, statusX, statusY + (lineSpacing * 2), lines[2], kStatusTextColor, textScale, 0, 0, false, false, false, false, 0, false, false);
}

static void DrawStatusScreen(uint64_t fps, size_t peopleCount)
{
    image_t statusImg;
    S_DISP_RECT statusRect;

    // Clear the active buffer region
    memset(frame_buf1, 0, RENDER_FB_SIZE);

    statusImg.w = RENDER_WIDTH;
    statusImg.h = RENDER_HEIGHT;
    statusImg.size = RENDER_FB_SIZE;
    statusImg.pixfmt = PIXFORMAT_RGB565;
    statusImg.data = (uint8_t *)frame_buf1;

    DrawStatusOverlay(&statusImg, fps, peopleCount);

#if USE_CCAP_CAMERA
    statusRect.u32TopLeftX = (LCD_DISPLAY_WIDTH - CCAP_CAPTURE_WIDTH * 2) / 2;     // 80
    statusRect.u32TopLeftY = (LCD_DISPLAY_HEIGHT - CCAP_CAPTURE_HEIGHT * 2) / 2;   // 0
    statusRect.u32BottonRightX = (LCD_DISPLAY_WIDTH - CCAP_CAPTURE_WIDTH * 2) / 2 + CCAP_CAPTURE_WIDTH * 2 - 1; // 719
    statusRect.u32BottonRightY = (LCD_DISPLAY_HEIGHT - CCAP_CAPTURE_HEIGHT * 2) / 2 + CCAP_CAPTURE_HEIGHT * 2 - 1; // 479
#else
    statusRect.u32TopLeftX = 0;
    statusRect.u32TopLeftY = 0;
    statusRect.u32BottonRightX = RENDER_WIDTH - 1;
    statusRect.u32BottonRightY = RENDER_HEIGHT - 1;
#endif

    // Clear the physical LCD screen to black first
    Display_ClearLCD(C_BLACK);

    // Draw the active window region
#if USE_CCAP_CAMERA
    Display_FillRect((uint16_t *)statusImg.data, &statusRect, 2);
#else
    Display_FillRect((uint16_t *)statusImg.data, &statusRect, 1);
#endif
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

#if !USE_CCAP_CAMERA
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
#endif /* !USE_CCAP_CAMERA */

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

/* --- LCD Display Task (EBI Blitting) --- */
#if defined(__EBI_LCD_PANEL__)
static void vDisplayTask(void *pvParameters)
{
    (void)pvParameters;
#if USE_CCAP_CAMERA
    // Centered 640x480 area on the 800x480 LCD panel (for 2x scale of 320x240)
    S_DISP_RECT sDispRect = {
        .u32TopLeftX = (LCD_DISPLAY_WIDTH - CCAP_CAPTURE_WIDTH * 2) / 2,     // 80
        .u32TopLeftY = (LCD_DISPLAY_HEIGHT - CCAP_CAPTURE_HEIGHT * 2) / 2,   // 0
        .u32BottonRightX = (LCD_DISPLAY_WIDTH - CCAP_CAPTURE_WIDTH * 2) / 2 + CCAP_CAPTURE_WIDTH * 2 - 1, // 719
        .u32BottonRightY = (LCD_DISPLAY_HEIGHT - CCAP_CAPTURE_HEIGHT * 2) / 2 + CCAP_CAPTURE_HEIGHT * 2 - 1 // 479
    };
#else
    S_DISP_RECT sDispRect = {
        .u32TopLeftX = 0,
        .u32TopLeftY = 0,
        .u32BottonRightX = RENDER_WIDTH - 1,
        .u32BottonRightY = RENDER_HEIGHT - 1
    };
#endif

    while (1)
    {
        /* Wait for notification from the inference task */
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        uint64_t t_start = pmu_get_systick_Count();
        /* Blit the ready show buffer to EBI screen (blocking PDMA/EBI write) */
#if USE_CCAP_CAMERA
        Display_FillRect((uint16_t *)g_lcdShowBuf, &sDispRect, 2);
#else
        Display_FillRect((uint16_t *)g_lcdShowBuf, &sDispRect, 1);
#endif
        uint64_t t_end = pmu_get_systick_Count();
        g_displayBlitCycles = (uint32_t)(t_end - t_start);

        /* Clear the pending flag so the next swap can occur */
        g_lcdBlitPending = false;
    }
}
#endif

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

    // Initialize the fast quantization lookup table using the exact model quantization parameters
    for (int val = 0; val < 256; ++val)
    {
        float normalized = static_cast<float>(val) / 255.0f;
        int32_t quantized = static_cast<int32_t>(roundf(normalized / inQuantParams.scale)) + inQuantParams.offset;
        if (quantized < -128) quantized = -128;
        if (quantized > 127)  quantized = 127;
        g_quantLUT[val] = static_cast<int8_t>(quantized);
    }

    arm::app::model::Detection detections[MODEL_MAX_DETECTIONS] = {};
    size_t detectionCount = 0;
    
    const bool useFastInputQuant =
        (IMAGE_CHANNELS == 3) &&
        (inQuantParams.offset == -1) &&
        (inQuantParams.scale > 0.007f) &&
        (inQuantParams.scale < 0.0085f);

    // OpenMV image struct for overlays/text on the LCD display buffer.
    image_t dstImg;
    dstImg.w = RENDER_WIDTH;
    dstImg.h = RENDER_HEIGHT;
    dstImg.size = RENDER_FB_SIZE;
    dstImg.pixfmt = PIXFORMAT_RGB565;
    dstImg.data = (uint8_t *)g_lcdDrawBuf;

#if defined(__EBI_LCD_PANEL__)
    S_DISP_RECT sDispRect;
#endif

    uint64_t frameCount = 0;
    uint64_t lastTime = pmu_get_systick_Count();
    uint64_t currentFPS = 0;

    // Profiling timing accumulators
    uint64_t sumCaptureWait = 0;
    uint64_t sumInputProc = 0;
    uint64_t sumInference = 0;
    uint64_t sumPostProcess = 0;
    uint64_t sumRenderPrep = 0;

    LOG_INFO("Inference Engine initialized. Stack high water mark: %u words remaining.", (unsigned int)uxTaskGetStackHighWaterMark(NULL));

#if USE_CCAP_CAMERA
    /* --- CCAP camera path: initialise sensor then capture in a tight loop --- */
    LOG_INFO("Initializing CCAP onboard camera (HM1055)...");
    if (ImageSensor_Init() != 0)
    {
        LOG_ERROR("ImageSensor_Init failed. Check CCAP wiring and sensor power.");
        vTaskDelete(NULL);
        return;
    }

    if (ImageSensor_Config(eIMAGE_FMT_RGB565, CCAP_CAPTURE_WIDTH, CCAP_CAPTURE_HEIGHT, true) != 0)
    {
        LOG_ERROR("ImageSensor_Config failed.");
        vTaskDelete(NULL);
        return;
    }
    LOG_INFO("CCAP camera ready. Capturing %ux%u RGB565 frames (pipeline mode).", CCAP_CAPTURE_WIDTH, CCAP_CAPTURE_HEIGHT);

#if defined(__EBI_LCD_PANEL__)
    /* In CCAP mode the network task is absent, so we initialise the LCD
     * from here.  Display_Delay() relies on the FreeRTOS PMU tick counter
     * which is now running because the scheduler has started. */
    LOG_INFO("Initializing LCD panel (CCAP mode)...");
    Display_Init();
    DrawStatusScreen(0, 0);
    LOG_INFO("LCD panel ready.");
#endif

    /* --- Two-buffer pipeline ---
     * While the NPU runs inference on the "ready" buffer, CCAP DMA fills the
     * "capture" buffer.  After inference we swap roles and immediately kick
     * the next capture so the sensor is never idle.
     *
     * captureBuf  — buffer currently being filled by CCAP DMA
     * readyBuf    — buffer whose previous capture is complete; safe to read
     */
    uint8_t *captureBuf = g_ccapBuf0;
    uint8_t *readyBuf   = g_ccapBuf1;

    /* Prime the pipeline: trigger the very first capture before entering loop */
    ImageSensor_TriggerCapture((uint32_t)captureBuf);

    /* imlib source image descriptor for the QVGA RGB565 capture buffer */
    image_t ccapSrcImg;
    ccapSrcImg.w      = CCAP_CAPTURE_WIDTH;
    ccapSrcImg.h      = CCAP_CAPTURE_HEIGHT;
    ccapSrcImg.size   = CCAP_FB_SIZE;
    ccapSrcImg.pixfmt = PIXFORMAT_RGB565;

    /* imlib destination: resize directly into the tensor input buffer as RGB888 */
    image_t tensorInputImg;
    tensorInputImg.w      = IMAGE_WIDTH;
    tensorInputImg.h      = IMAGE_HEIGHT;
    tensorInputImg.size   = FRAME_BUFFER_SIZE;
    tensorInputImg.pixfmt = PIXFORMAT_RGB888;
    tensorInputImg.data   = (uint8_t *)inputTensor->data.data;

    rectangle_t captureRoi;
    captureRoi.x = 0;
    captureRoi.y = 0;
    captureRoi.w = CCAP_CAPTURE_WIDTH;
    captureRoi.h = CCAP_CAPTURE_HEIGHT;

    while (1)
    {
        /* --- Step A: wait for CCAP DMA into captureBuf to finish --- */
        uint64_t t_start_wait = pmu_get_systick_Count();
        ImageSensor_WaitCaptureDone();
        uint64_t t_end_wait = pmu_get_systick_Count();
        sumCaptureWait += (t_end_wait - t_start_wait);

        /* --- Step B: swap buffers --- */
        uint8_t *tmp = readyBuf;
        readyBuf     = captureBuf;
        captureBuf   = tmp;

        /* --- Step C: immediately trigger next capture into the new captureBuf
         *             so CCAP DMA and NPU inference run concurrently.        --- */
        ImageSensor_TriggerCapture((uint32_t)captureBuf);

        /* --- Step D: resize + convert readyBuf (RGB565 QVGA) directly into
         *             the tensor input buffer (RGB888 192x192).              --- */
        uint64_t t_start_inproc = pmu_get_systick_Count();
        ccapSrcImg.data = readyBuf;
        imlib_nvt_scale(&ccapSrcImg, &tensorInputImg, &captureRoi);

        /* --- Step E: LUT fast in-place quantization (uint8 -> int8) --- */
        {
            uint8_t  *u8  = (uint8_t  *)inputTensor->data.data;
            int8_t   *s8  = (int8_t   *)inputTensor->data.data;
            for (int i = 0; i < FRAME_BUFFER_SIZE; ++i)
            {
                s8[i] = g_quantLUT[u8[i]];
            }
        }
        uint64_t t_end_inproc = pmu_get_systick_Count();
        sumInputProc += (t_end_inproc - t_start_inproc);

        /* inferenceFrame pointer used by the LCD display path below */
        const uint8_t *inferenceFrame = readyBuf;  // RGB565 QVGA
#else
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
#endif /* USE_CCAP_CAMERA */

#if defined(__EBI_LCD_PANEL__)
        dstImg.data = (uint8_t *)g_lcdDrawBuf;
#endif

#if !USE_CCAP_CAMERA
        // 1. Quantize the raw RGB pixels into the model input tensor (int8)
        uint64_t t_start_inproc = pmu_get_systick_Count();
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
                signedInputData[i] = g_quantLUT[inferenceFrame[i]];
            }
        }
        uint64_t t_end_inproc = pmu_get_systick_Count();
        sumInputProc += (t_end_inproc - t_start_inproc);
#endif /* !USE_CCAP_CAMERA */

        // 2. Execute Ethos-U Accelerated Inference
        uint64_t t_start_inf = pmu_get_systick_Count();
        model.RunInference();
        uint64_t t_end_inf = pmu_get_systick_Count();
        sumInference += (t_end_inf - t_start_inf);

        // 3. Post-Process Grid heatmaps to get target peaks (person locations)
        uint64_t t_start_post = pmu_get_systick_Count();
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
        uint64_t t_end_post = pmu_get_systick_Count();
        sumPostProcess += (t_end_post - t_start_post);

        // 4. Update Metrics (FPS and Person Counts)
        frameCount++;
        uint64_t now = pmu_get_systick_Count();
        if (now - lastTime >= SystemCoreClock * 5) // 5 second interval
        {
            currentFPS = frameCount / 5;
            uint64_t divisor = frameCount > 0 ? frameCount : 1;
            frameCount = 0;
            
            float to_ms = 1000.0f / SystemCoreClock;
            
            LOG_INFO("[STATUS] FPS: %llu | People: %d", currentFPS, (int)detectionCount);
            LOG_INFO("[PROFILE] Avg times per frame:");
            LOG_INFO("  - Camera Wait:   %6.2f ms", (static_cast<float>(sumCaptureWait) / divisor) * to_ms);
            LOG_INFO("  - Input Scale/Q: %6.2f ms", (static_cast<float>(sumInputProc) / divisor) * to_ms);
            LOG_INFO("  - Inference:     %6.2f ms", (static_cast<float>(sumInference) / divisor) * to_ms);
            LOG_INFO("  - Post-Process:  %6.2f ms", (static_cast<float>(sumPostProcess) / divisor) * to_ms);
            LOG_INFO("  - Render Prep:   %6.2f ms", (static_cast<float>(sumRenderPrep) / divisor) * to_ms);
            LOG_INFO("  - EBI Blit (HW): %6.2f ms", static_cast<float>(g_displayBlitCycles) * to_ms);
            
            sumCaptureWait = 0;
            sumInputProc = 0;
            sumInference = 0;
            sumPostProcess = 0;
            sumRenderPrep = 0;
            lastTime = now;
        }

        // 5. Visual Rendering on LCD Panel (if enabled)
#if defined(__EBI_LCD_PANEL__)
        uint64_t t_start_render = pmu_get_systick_Count();
#if USE_CCAP_CAMERA
        // 5a. Draw crosshair indicators directly on the fast QVGA SRAM image first
        for (size_t i = 0; i < detectionCount; ++i)
        {
            const arm::app::model::Detection& det = detections[i];
            int x_disp = static_cast<int>((det.x * CCAP_CAPTURE_WIDTH) / IMAGE_WIDTH);
            int y_disp = static_cast<int>((det.y * CCAP_CAPTURE_HEIGHT) / IMAGE_HEIGHT);
            int box_w = CCAP_CAPTURE_WIDTH / 24;
            int box_h = CCAP_CAPTURE_HEIGHT / 24;
            if (box_w < 8) box_w = 8;
            if (box_h < 8) box_h = 8;

            // Draw a bounding crosshair directly on CCAP source image (in fast SRAM2)
            imlib_draw_rectangle(&ccapSrcImg,
                                 x_disp - (box_w / 2),
                                 y_disp - (box_h / 2),
                                 box_w,
                                 box_h,
                                 COLOR_R5_G6_B5_TO_RGB565(31, 0, 0),
                                 2,
                                 false);
            
            // Draw center dot
            imlib_draw_rectangle(&ccapSrcImg, x_disp - 1, y_disp - 1, 2, 2, COLOR_R5_G6_B5_TO_RGB565(31, 31, 0), 1, true);
        }

        DrawStatusOverlay(&ccapSrcImg, currentFPS, detectionCount);

        // 5b. Direct copy of the SRAM image (with overlays) to HyperRAM
        memcpy(dstImg.data, inferenceFrame, CCAP_FB_SIZE);
#else
        // UDP path: inferenceFrame is RGB888 192×192 — scale and convert to RGB565.
        ConvertRgb888ToRgb565Scaled(inferenceFrame,
                                    (uint16_t *)dstImg.data,
                                    LCD_DISPLAY_WIDTH,
                                    LCD_DISPLAY_HEIGHT);

        // Draw crosshair indicators at each person peak
        for (size_t i = 0; i < detectionCount; ++i)
        {
            const arm::app::model::Detection& det = detections[i];
            int x_disp = static_cast<int>((det.x * RENDER_WIDTH) / IMAGE_WIDTH);
            int y_disp = static_cast<int>((det.y * RENDER_HEIGHT) / IMAGE_HEIGHT);
            int box_w = RENDER_WIDTH / 24;
            int box_h = RENDER_HEIGHT / 24;
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
#endif /* USE_CCAP_CAMERA */
        uint64_t t_end_render = pmu_get_systick_Count();
        sumRenderPrep += (t_end_render - t_start_render);

        // Blit to screen using the background display task (double-buffered)
        if (!g_lcdBlitPending)
        {
            g_lcdBlitPending = true;

            // Swap draw buffer and show buffer
            char *tmp = g_lcdShowBuf;
            g_lcdShowBuf = g_lcdDrawBuf;
            g_lcdDrawBuf = tmp;

            // Notify display task to start background blit
            if (xDisplayTaskHandle != NULL)
            {
                xTaskNotifyGive(xDisplayTaskHandle);
            }
        }
#endif
#if !USE_CCAP_CAMERA
        taskENTER_CRITICAL();
        if (g_processingFrameBufferIndex == frameBufferIndex)
        {
            g_processingFrameBufferIndex = -1;
        }
        taskEXIT_CRITICAL();
#endif /* !USE_CCAP_CAMERA */
    }
}

#if !USE_CCAP_CAMERA
/* LwIP TCP/IP stack thread task (UDP server path only) */
static void vNetworkInitTask(void *pvParameters)
{
    (void)pvParameters;
    ip_addr_t ipaddr, netmask, gw;
    struct netif *netifResult;

#if defined(__EBI_LCD_PANEL__)
    // Display_Init() eventually calls Display_Delay(), which waits on the
    // PMU/SysTick-derived counter updated from the FreeRTOS tick hook. Keep LCD
    // initialization inside a task, after vTaskStartScheduler() has started.
    // Calling Display_Init() from main() before task creation/scheduler start can
    // hang forever in Display_Delay() because the counter may not advance yet.
    //
    // The first draw happens before network setup so the LCD is useful even when
    // no UDP feed is present. The IP-specific redraw below must remain after lwIP
    // assigns/configures g_netif.ip_addr.
    LOG_INFO("Initializing LCD panel...");
    Display_Init();
    LOG_INFO("LCD panel initialization returned. Drawing status screen...");
    DrawStatusScreen(0, 0);
    LOG_INFO("LCD status screen drawn.");
#endif

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
    // Redraw the status screen after the network address is known. This is the
    // no-feed discovery path: users can read the device IP even before sending a
    // UDP video stream. Keep this after DHCP/static netif configuration and before
    // the NetInit task suspends.
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
#endif /* !USE_CCAP_CAMERA */

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
                         eMPU_ATTR_CACHEABLE_WBWARA) // Cacheable Write-Back
        },
        {
            ARM_MPU_RBAR(((unsigned int)frame_buf1),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Only
                         0,                 // Non-Privileged
                         1),                // eXecute Never
            ARM_MPU_RLAR((((unsigned int)frame_buf1) + sizeof(frame_buf1) - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WBWARA) // Cacheable Write-Back
        },
        {
            ARM_MPU_RBAR(((unsigned int)frame_buf2),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Only
                         0,                 // Non-Privileged
                         1),                // eXecute Never
            ARM_MPU_RLAR((((unsigned int)frame_buf2) + sizeof(frame_buf2) - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WBWARA) // Cacheable Write-Back
        }
    };

    // Apply custom MPU regions
    LOG_INFO("Configuring custom MPU memory caching regions...");
    InitPreDefMPURegion(&mpuConfig[0], sizeof(mpuConfig) / sizeof(mpuConfig[0]));
    LOG_INFO("Custom MPU memory cache regions applied successfully (tensorArena WTRA, framebuffers cacheable WBWARA).");

    // Enable MemManage, BusFault, and UsageFault handlers explicitly in SCB
    LOG_INFO("Enabling dedicated SCB fault handlers (MemManage, BusFault, UsageFault)...");
    SCB->SHCSR |= (SCB_SHCSR_MEMFAULTENA_Msk | SCB_SHCSR_BUSFAULTENA_Msk | SCB_SHCSR_USGFAULTENA_Msk);
    LOG_INFO("SCB fault handlers activated successfully.");

    // Initialize OpenMV memory allocators before any status or camera rendering.
    // LCD initialization is deliberately not done here: the LCD driver delay path
    // depends on the FreeRTOS tick/PMU counter, so it must run from a task after
    // vTaskStartScheduler() begins.
    LOG_INFO("Initializing OpenMV frame buffer allocators...");
    omv_init();
    LOG_INFO("OpenMV frame buffer memory allocator initialized successfully.");

#if USE_CCAP_CAMERA
    LOG_INFO("----------------------------------------------------------------");
    LOG_INFO("    Starting M55M1 CCAP Camera People Counting Firmware    ");
    LOG_INFO("----------------------------------------------------------------");
#else
    LOG_INFO("----------------------------------------------------------------");
    LOG_INFO("    Starting M55M1 UDP Server People Counting Firmware     ");
    LOG_INFO("----------------------------------------------------------------");
#endif

#if !USE_CCAP_CAMERA
    // Create NetworkInit task only in UDP server mode
    LOG_INFO("Spawning NetworkInit FreeRTOS task (Stack: 2048 words, Priority: %lu)...", (unsigned long)(tskIDLE_PRIORITY + 4UL));
    if (xTaskCreate(vNetworkInitTask, "NetInit", 2048, NULL, tskIDLE_PRIORITY + 4UL, NULL) != pdPASS)
    {
        LOG_ERROR("Failed to create NetworkInit task. FreeRTOS heap remaining: %u bytes.",
                  (unsigned int)xPortGetFreeHeapSize());
        while (1);
    }
#endif /* !USE_CCAP_CAMERA */

#if defined(__EBI_LCD_PANEL__)
    LOG_INFO("Spawning LCD Display FreeRTOS task (Stack: 1024 words, Priority: %lu)...", (unsigned long)(tskIDLE_PRIORITY + 3UL));
    if (xTaskCreate(vDisplayTask, "Display", 1024, NULL, tskIDLE_PRIORITY + 3UL, &xDisplayTaskHandle) != pdPASS)
    {
        LOG_ERROR("Failed to create LCD Display task. FreeRTOS heap remaining: %u bytes.",
                  (unsigned int)xPortGetFreeHeapSize());
        while (1);
    }
#endif

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
