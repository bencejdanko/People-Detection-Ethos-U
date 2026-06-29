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



/* Board and BSP includes */
#include "NuMicro.h"
#include "BoardInit.hpp"
#include "board_config.h"
#include "pmu_counter.h"

/* Global variables and TrustZone stub symbols for non-secure FreeRTOS */
extern "C" {
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
#include "ff.h"
#include "ModelFileReader.h"
#include "InferenceModel.hpp"
#include "PostProcessor.hpp"
#include "wifi_push.hpp"

#if defined(__EBI_LCD_PANEL__)
#include "Display.h"
#endif

/* Task Handles */
static TaskHandle_t xInferenceTaskHandle = NULL;

/* --- CCAP FRAME BUFFER MEMORY ALLOCATION ---
 * Two QVGA (320x240) RGB565 ping-pong buffers so that CCAP DMA into one
 * buffer can overlap with NPU inference on the other. */
#define CCAP_CAPTURE_WIDTH   320
#define CCAP_CAPTURE_HEIGHT  240
#define CCAP_FB_SIZE         (CCAP_CAPTURE_WIDTH * CCAP_CAPTURE_HEIGHT * 2)  // RGB565
__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t g_ccapBuf0[CCAP_FB_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t g_ccapBuf1[CCAP_FB_SIZE];

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

#define RENDER_WIDTH   CCAP_CAPTURE_WIDTH
#define RENDER_HEIGHT  CCAP_CAPTURE_HEIGHT
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



#if defined(__EBI_LCD_PANEL__)
static const int kStatusTextScale = 4;
static const int kStatusTextMargin = 16;
static const int kStatusTextLineHeight = FONT_HTIGHT * kStatusTextScale + 8;
static const int kStatusTextColor = COLOR_R5_G6_B5_TO_RGB565(31, 63, 31);

static void DrawStatusOverlay(image_t *img, uint64_t fps, size_t peopleCount)
{
    char lines[4][40];
    int textScale = (img->h >= 480) ? 4 : 2;
    int lineSpacing = FONT_HTIGHT * textScale + 8;
    const int statusX = kStatusTextMargin;
    const int statusY = img->h - kStatusTextMargin - (lineSpacing * 4);

    sprintf(lines[0], "FPS: %llu", fps);
    sprintf(lines[1], "PEOPLE: %d", (int)peopleCount);
    sprintf(lines[2], "SRC: CCAP Camera");

    bool wifiConnected = WifiPush::IsConnected();
    sprintf(lines[3], "WIFI: %s", wifiConnected ? "Connected" : "Disconnected");
    uint16_t wifiColor = wifiConnected ? COLOR_R5_G6_B5_TO_RGB565(0, 63, 0) : COLOR_R5_G6_B5_TO_RGB565(31, 0, 0);

    imlib_draw_string(img, statusX, statusY, lines[0], kStatusTextColor, textScale, 0, 0, false, false, false, false, 0, false, false);
    imlib_draw_string(img, statusX, statusY + lineSpacing, lines[1], kStatusTextColor, textScale, 0, 0, false, false, false, false, 0, false, false);
    imlib_draw_string(img, statusX, statusY + (lineSpacing * 2), lines[2], kStatusTextColor, textScale, 0, 0, false, false, false, false, 0, false, false);
    imlib_draw_string(img, statusX, statusY + (lineSpacing * 3), lines[3], wifiColor, textScale, 0, 0, false, false, false, false, 0, false, false);
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

    statusRect.u32TopLeftX = (LCD_DISPLAY_WIDTH - CCAP_CAPTURE_WIDTH * 2) / 2;     // 80
    statusRect.u32TopLeftY = (LCD_DISPLAY_HEIGHT - CCAP_CAPTURE_HEIGHT * 2) / 2;   // 0
    statusRect.u32BottonRightX = (LCD_DISPLAY_WIDTH - CCAP_CAPTURE_WIDTH * 2) / 2 + CCAP_CAPTURE_WIDTH * 2 - 1; // 719
    statusRect.u32BottonRightY = (LCD_DISPLAY_HEIGHT - CCAP_CAPTURE_HEIGHT * 2) / 2 + CCAP_CAPTURE_HEIGHT * 2 - 1; // 479

    // Clear the physical LCD screen to black first
    Display_ClearLCD(C_BLACK);

    // Draw the active window region
    Display_FillRect((uint16_t *)statusImg.data, &statusRect, 2);
}
#endif



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



static bool ModelContainsEthosUCustomOp(const unsigned char *modelData, size_t modelSize)
{
    static const char ethosUOpName[] = "ethos-u";
    const size_t opNameLen = sizeof(ethosUOpName) - 1;

    if (modelSize < opNameLen) {
        return false;
    }

    for (unsigned int i = 0; i <= modelSize - opNameLen; ++i) {
        if (std::memcmp(&modelData[i], ethosUOpName, opNameLen) == 0) {
            return true;
        }
    }

    return false;
}

#define MODEL_AT_HYPERRAM_ADDR (0x82400000)

static int32_t LoadModelFromSDCard(const unsigned char **modelData)
{
#define MODEL_FILE "0:\\MODEL.TFL"
#define EACH_READ_SIZE 4096
	
    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */	
    f_chdrive(sd_path);          /* set default path */

	int32_t i32FileSize;
	int32_t i32FileReadIndex = 0;
	int32_t i32Read;
	
	if(!ModelFileReader_Initialize(MODEL_FILE))
	{
        LOG_ERROR("Unable to open model file %s on SD card.", MODEL_FILE);		
		return -1;
	}
	
	i32FileSize = ModelFileReader_FileSize();
    LOG_INFO("Model file size: %d bytes.", (int)i32FileSize);

	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		i32FileReadIndex += i32Read;
	}
	
	if(i32FileReadIndex < i32FileSize)
	{
        LOG_ERROR("Read Model file size is not enough (expected %d, read %d).", (int)i32FileSize, (int)i32FileReadIndex);		
		return -2;
	}
	
	ModelFileReader_Finish();
	*modelData = (const unsigned char *)MODEL_AT_HYPERRAM_ADDR;
	return i32FileSize;
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
#endif // __EBI_LCD_PANEL__

#if USE_CCAP_CAMERA
static void ScaleQuantizeCcapYolo(const uint16_t *src, int8_t *dst, const int8_t *lut)
{
    // Pad top 24 rows
    int8_t padVal = lut[0];
    memset(dst, padVal, 24 * 192 * 3);

    // Scale and quantize center 144 rows
    for (int y = 0; y < 144; ++y) {
        int src_y = (y * 5) / 3;
        const uint16_t *srcRow = src + src_y * CCAP_CAPTURE_WIDTH;
        int8_t *dstRow = dst + (24 + y) * 192 * 3;

        for (int x = 0; x < 192; ++x) {
            int src_x = (x * 5) / 3;
            uint16_t p = srcRow[src_x];

            uint8_t r = ((p >> 11) & 0x1F) << 3;
            uint8_t g = ((p >> 5) & 0x3F) << 2;
            uint8_t b = (p & 0x1F) << 3;

            int dst_idx = x * 3;
            dstRow[dst_idx + 0] = lut[r];
            dstRow[dst_idx + 1] = lut[g];
            dstRow[dst_idx + 2] = lut[b];
        }
    }

    // Pad bottom 24 rows
    memset(dst + (168 * 192 * 3), padVal, 24 * 192 * 3);
}
#endif

/* --- ML Inference and Post-Processing Task --- */
static void vInferenceTask(void *pvParameters)
{
    (void)pvParameters;
    
    // Use model from SD card.
    const unsigned char *modelData = NULL;
    int32_t modelSize = LoadModelFromSDCard(&modelData);
    if (modelSize <= 0) {
        LOG_ERROR("Failed to load model from SD card.");
        vTaskDelete(NULL);
        return;
    }

    if (!ModelContainsEthosUCustomOp(modelData, modelSize)) {
        LOG_ERROR("Model is not Vela-optimized for Ethos-U (missing custom op \"ethos-u\").");
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
        MODEL_INPUT_HEIGHT
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
        float normalized = static_cast<float>(val);
        if (inQuantParams.scale < 0.05f)
        {
            if (inQuantParams.offset > -100)
            {
                normalized = (normalized - 127.5f) / 127.5f;
            }
            else
            {
                normalized = normalized / 255.0f;
            }
        }
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

    // Input scaling and quantization variables are local or static

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

        /* --- Step D: resize and quantize readyBuf (RGB565 QVGA) directly into
         *             the tensor input buffer (int8 192x192 RGB888).         --- */
        uint64_t t_start_inproc = pmu_get_systick_Count();
        ccapSrcImg.data = readyBuf;
        ScaleQuantizeCcapYolo((const uint16_t *)readyBuf, inputTensor->data.int8, g_quantLUT);
        uint64_t t_end_inproc = pmu_get_systick_Count();
        sumInputProc += (t_end_inproc - t_start_inproc);

        /* inferenceFrame pointer used by the LCD display path below */
        const uint8_t *inferenceFrame = readyBuf;  // RGB565 QVGA


#if defined(__EBI_LCD_PANEL__)
        dstImg.data = (uint8_t *)g_lcdDrawBuf;
#endif



        // 2. Execute Ethos-U Accelerated Inference
        uint64_t t_start_inf = pmu_get_systick_Count();
        model.RunInference();
        uint64_t t_end_inf = pmu_get_systick_Count();
        sumInference += (t_end_inf - t_start_inf);

        // 3. Post-Process Grid heatmaps to get target peaks (person locations)
        uint64_t t_start_post = pmu_get_systick_Count();
        const int8_t *outputData = outputTensor->data.int8;
        postProcessor.Process(
            &model,
            MODEL_DEFAULT_THRESHOLD,
            detections,
            MODEL_MAX_DETECTIONS,
            detectionCount
        );
        uint64_t t_end_post = pmu_get_systick_Count();
        sumPostProcess += (t_end_post - t_start_post);

        // Calculate and push count of detected people (class 0)
        size_t wifiPersonCount = 0;
        for (size_t i = 0; i < detectionCount; ++i)
        {
            if (detections[i].cls == 0)
            {
                wifiPersonCount++;
            }
        }
        WifiPush::PushCount(static_cast<int>(wifiPersonCount));

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
        // 5a. Draw YOLOv8n bounding boxes directly on the fast QVGA SRAM image first
        size_t personCount = 0;
        for (size_t i = 0; i < detectionCount; ++i)
        {
            const arm::app::model::Detection& det = detections[i];
            if (det.cls != 0) continue; // Keep only class 0 (person)
            personCount++;
            
            // Map back to CCAP coordinate space (undoing 24px letterbox pad)
            float x_scaled = det.x;
            float y_scaled = det.y - 24.0f;
            
            int x_disp = static_cast<int>(x_scaled * 5.0f / 3.0f);
            int y_disp = static_cast<int>(y_scaled * 5.0f / 3.0f);
            int w_disp = static_cast<int>(det.w * 5.0f / 3.0f);
            int h_disp = static_cast<int>(det.h * 5.0f / 3.0f);

            // Clip coordinates to screen bounds (320x240)
            if (x_disp < 0) { w_disp += x_disp; x_disp = 0; }
            if (y_disp < 0) { h_disp += y_disp; y_disp = 0; }
            if (x_disp + w_disp > CCAP_CAPTURE_WIDTH) { w_disp = CCAP_CAPTURE_WIDTH - x_disp; }
            if (y_disp + h_disp > CCAP_CAPTURE_HEIGHT) { h_disp = CCAP_CAPTURE_HEIGHT - y_disp; }

            if (w_disp > 0 && h_disp > 0)
            {
                imlib_draw_rectangle(&ccapSrcImg,
                                     x_disp,
                                     y_disp,
                                     w_disp,
                                     h_disp,
                                     COLOR_R5_G6_B5_TO_RGB565(0, 63, 0), // Clean Green Box
                                     2,
                                     false);
            }
        }

        DrawStatusOverlay(&ccapSrcImg, currentFPS, personCount);

        // 5b. Direct copy of the SRAM image (with overlays) to HyperRAM
        memcpy(dstImg.data, inferenceFrame, CCAP_FB_SIZE);
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

    }
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

    LOG_INFO("----------------------------------------------------------------");
    LOG_INFO("    Starting M55M1 CCAP Camera People Counting Firmware    ");
    LOG_INFO("----------------------------------------------------------------");

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

    // Configure and start Wi-Fi pushes
    WifiPush::Configure(WIFI_SSID, WIFI_PASS, SERVER_HOST, SERVER_PORT, SERVER_PATH);
    WifiPush::Start();

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
