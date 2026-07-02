/**************************************************************************//**
 * @file     PostProcessor.cpp
 * @version  V1.00
 * @brief    C++ post-processing implementation for YOLO26 NMS-free object detection
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "PostProcessor.hpp"
#include <algorithm>

namespace arm
{
namespace app
{
namespace model
{

PostProcessor::PostProcessor(int inputWidth, int inputHeight)
    : m_inputWidth(inputWidth),
      m_inputHeight(inputHeight)
{
    m_detections.reserve(128);
}

void PostProcessor::GetNetworkBoxes(arm::app::Model* model, std::vector<Detection>& detections, float threshold)
{
    TfLiteTensor* psOutputTensor = model->GetOutputTensor(0);
    if (!psOutputTensor)
    {
        return;
    }

    if (psOutputTensor->dims->size < 3)
    {
        return;
    }

    int dim1 = psOutputTensor->dims->data[1]; // channels (5) or anchors (756)
    int dim2 = psOutputTensor->dims->data[2]; // anchors (756) or channels (5)
    int totalAnchors = 0;
    bool channelsFirst = true;

    if (dim1 == 5)
    {
        totalAnchors = dim2;
        channelsFirst = true;
    }
    else if (dim2 == 5)
    {
        totalAnchors = dim1;
        channelsFirst = false;
    }
    else
    {
        return;
    }

    float scale = 1.0f;
    int zeroPoint = 0;
    if (psOutputTensor->quantization.type == kTfLiteAffineQuantization)
    {
        TfLiteAffineQuantization* quantParams = (TfLiteAffineQuantization*)psOutputTensor->quantization.params;
        if (quantParams && quantParams->scale && quantParams->scale->size > 0)
        {
            scale = quantParams->scale->data[0];
            zeroPoint = quantParams->zero_point->data[0];
        }
    }

    int8_t* tensorData = psOutputTensor->data.int8;

    for (int i = 0; i < totalAnchors; i++)
    {
        int8_t x1_quant, y1_quant, x2_quant, y2_quant, score_quant;
        if (channelsFirst)
        {
            x1_quant    = tensorData[0 * totalAnchors + i];
            y1_quant    = tensorData[1 * totalAnchors + i];
            x2_quant    = tensorData[2 * totalAnchors + i];
            y2_quant    = tensorData[3 * totalAnchors + i];
            score_quant = tensorData[4 * totalAnchors + i];
        }
        else
        {
            x1_quant    = tensorData[i * 5 + 0];
            y1_quant    = tensorData[i * 5 + 1];
            x2_quant    = tensorData[i * 5 + 2];
            y2_quant    = tensorData[i * 5 + 3];
            score_quant = tensorData[i * 5 + 4];
        }

        // Dequantize score
        float score = scale * (static_cast<float>(score_quant) - zeroPoint);

        if (score >= threshold)
        {
            // Dequantize coordinates (which are in input MODEL_INPUT_WIDTH x MODEL_INPUT_HEIGHT pixel coordinate space)
            float x1 = scale * (static_cast<float>(x1_quant) - zeroPoint);
            float y1 = scale * (static_cast<float>(y1_quant) - zeroPoint);
            float x2 = scale * (static_cast<float>(x2_quant) - zeroPoint);
            float y2 = scale * (static_cast<float>(y2_quant) - zeroPoint);

            Detection det;
            det.cls = 0; // 'person'
            det.score = score;
            det.x = x1;
            det.y = y1;
            det.w = x2 - x1;
            det.h = y2 - y1;

            detections.push_back(det);
        }
    }
}

void PostProcessor::Process(
    arm::app::Model* model,
    float threshold,
    Detection* results,
    size_t maxResults,
    size_t& resultCount
)
{
    m_detections.clear();
    GetNetworkBoxes(model, m_detections, threshold);

    // Sort detections by score descending to get highest confidence first
    std::sort(m_detections.begin(), m_detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    resultCount = 0;
    for (const auto& det : m_detections)
    {
        if (resultCount >= maxResults)
        {
            break;
        }

        // Clip box boundaries to input width/height space
        float x = std::min(std::max(det.x, 0.0f), static_cast<float>(m_inputWidth - 1));
        float y = std::min(std::max(det.y, 0.0f), static_cast<float>(m_inputHeight - 1));
        float w = std::min(std::max(det.w, 0.0f), static_cast<float>(m_inputWidth - 1));
        float h = std::min(std::max(det.h, 0.0f), static_cast<float>(m_inputHeight - 1));

        results[resultCount++] = {x, y, w, h, det.score, det.cls};
    }
}

} /* namespace model */
} /* namespace app */
} /* namespace arm */
