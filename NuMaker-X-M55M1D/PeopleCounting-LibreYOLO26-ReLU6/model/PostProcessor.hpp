/**************************************************************************//**
 * @file     PostProcessor.hpp
 * @version  V1.00
 * @brief    C++ post-processing for YOLO26 NMS-free object detection
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __POST_PROCESSOR_HPP__
#define __POST_PROCESSOR_HPP__

#include <cstdint>
#include <stddef.h>
#include <vector>
#include "Model.hpp"

namespace arm
{
namespace app
{
namespace model
{

struct Detection
{
    float x;       // Bounding box top-left X in 192x192 space
    float y;       // Bounding box top-left Y in 192x192 space
    float w;       // Bounding box width
    float h;       // Bounding box height
    float score;   // Confidence score (0.0 to 1.0)
    int cls;       // Class index (0 for person)
};

class PostProcessor
{
public:
    PostProcessor(int inputWidth, int inputHeight);
    ~PostProcessor() = default;

    /**
     * @brief Run post-processing on raw output of YOLO26 model
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
    std::vector<Detection> m_detections;

    void GetNetworkBoxes(arm::app::Model* model, std::vector<Detection>& detections, float threshold);
};

} /* namespace model */
} /* namespace app */
} /* namespace arm */

#endif // __POST_PROCESSOR_HPP__
