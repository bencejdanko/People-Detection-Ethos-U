/**************************************************************************//**
 * @file     PostProcessor.hpp
 * @version  V1.00
 * @brief    C++ post-processing for YOLOv8n object detection
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __POST_PROCESSOR_HPP__
#define __POST_PROCESSOR_HPP__

#include <cstdint>
#include <stddef.h>
#include <vector>
#include <forward_list>
#include "Model.hpp"

namespace arm
{
namespace app
{
namespace model
{

#define YOLOV8N_OD_STRIDE_8    (8)
#define YOLOV8N_OD_STRIDE_16   (16)
#define YOLOV8N_OD_STRIDE_32   (32)

// Vela compiled output tensor indices
#define YOLOV8N_OD_STRIDE8_CONFIDENCE_TENSOR_INDEX   (2)     // [1, 576, 80]
#define YOLOV8N_OD_STRIDE16_CONFIDENCE_TENSOR_INDEX  (4)     // [1, 144, 80]
#define YOLOV8N_OD_STRIDE32_CONFIDENCE_TENSOR_INDEX  (3)     // [1, 36, 80]

#define YOLOV8N_OD_STRIDE8_BOX_TENSOR_INDEX          (0)     // [1, 576, 64]
#define YOLOV8N_OD_STRIDE16_BOX_TENSOR_INDEX         (1)     // [1, 144, 64]
#define YOLOV8N_OD_STRIDE32_BOX_TENSOR_INDEX         (5)     // [1, 36, 64]

#define YOLOV8N_OD_CLASS       (1)     // Person-only class

struct Detection
{
    float x;       // Bounding box top-left X in 192x192 space
    float y;       // Bounding box top-left Y in 192x192 space
    float w;       // Bounding box width
    float h;       // Bounding box height
    float score;   // Confidence score (0.0 to 1.0)
    int cls;       // Class index (0 for person)
};

struct AnchorBox
{
    float w;
    float h;
};

class PostProcessor
{
public:
    PostProcessor(int inputWidth, int inputHeight);
    ~PostProcessor() = default;

    /**
     * @brief Run post-processing on raw outputs of YOLOv8n model
     * @param model Pointer to the base Model class
     * @param threshold Confidence threshold (0.0 to 1.0)
     * @param results Fixed-size detection buffer to populate
     * @param maxResults Maximum number of detections that fit in results
     * @param resultCount Number of detections written to results
     */
    void Process(arm::app::Model* model,
                 float threshold,
                 Detection* results,
                 size_t maxResults,
                 size_t& resultCount);

private:
    int m_inputWidth;
    int m_inputHeight;
    int m_stride8_total_anchors;
    int m_stride16_total_anchors;
    int m_stride32_total_anchors;

    std::vector<AnchorBox> m_stride8_anchors;
    std::vector<AnchorBox> m_stride16_anchors;
    std::vector<AnchorBox> m_stride32_anchors;
    std::vector<float> m_softmaxBuf;
    std::vector<Detection> m_detections;

    void GetNetworkBoxes(arm::app::Model* model, std::vector<Detection>& detections, float threshold);
    void CalDetectionBox(TfLiteTensor* psConfidenceOutputTensor,
                          TfLiteTensor* psBoxOutputTensor,
                          std::vector<AnchorBox>& vAnchorBoxes,
                          int stride,
                          int totalAnchors,
                          float threshold,
                          std::vector<Detection>& detections);
    void CalBoxXYWH(TfLiteTensor* psBoxOutputTensor,
                    std::vector<AnchorBox>& vAnchorBoxes,
                    int anchorIndex,
                    int stride,
                    int totalAnchors,
                    Detection& det);
    void CalculateNMS(std::vector<Detection>& detections, float iouThreshold);
};

} /* namespace model */
} /* namespace app */
} /* namespace arm */

#endif // __POST_PROCESSOR_HPP__
