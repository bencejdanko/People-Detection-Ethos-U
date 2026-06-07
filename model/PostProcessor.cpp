/**************************************************************************//**
 * @file     PostProcessor.cpp
 * @version  V1.00
 * @brief    C++ post-processing implementation for YOLOv8n object detection
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "PostProcessor.hpp"
#include "PlatformMath.hpp"
#include <cmath>
#include <algorithm>

namespace arm
{
namespace app
{
namespace model
{

static void AnchorMatrixConstruct(
    std::vector<AnchorBox>& vAnchorBoxes,
    int stride,
    int totalAnchors,
    int inputWidth
)
{
    float fStartAnchorValue = 0.5f;
    int iMaxAnchorValue = (inputWidth / stride);
    float fAnchor0StepValue = 0.0f;
    float fAnchor1StepValue = -1.0f;

    for (int i = 0; i < totalAnchors; i++)
    {
        AnchorBox sAnchorBox;

        if ((i % iMaxAnchorValue) == 0)
        {
            fStartAnchorValue = 0.5f;
            fAnchor0StepValue = 0.0f;
            fAnchor1StepValue++;
        }

        sAnchorBox.w = fStartAnchorValue + (fAnchor0StepValue++);
        sAnchorBox.h = fStartAnchorValue + fAnchor1StepValue;
        
        vAnchorBoxes.push_back(sAnchorBox);
    }
}

PostProcessor::PostProcessor(int inputWidth, int inputHeight)
    : m_inputWidth(inputWidth),
      m_inputHeight(inputHeight)
{
    m_stride8_total_anchors = (m_inputWidth / YOLOV8N_OD_STRIDE_8) * (m_inputWidth / YOLOV8N_OD_STRIDE_8);
    m_stride16_total_anchors = (m_inputWidth / YOLOV8N_OD_STRIDE_16) * (m_inputWidth / YOLOV8N_OD_STRIDE_16);
    m_stride32_total_anchors = (m_inputWidth / YOLOV8N_OD_STRIDE_32) * (m_inputWidth / YOLOV8N_OD_STRIDE_32);

    m_stride8_anchors.clear();
    m_stride16_anchors.clear();
    m_stride32_anchors.clear();

    AnchorMatrixConstruct(m_stride8_anchors, YOLOV8N_OD_STRIDE_8, m_stride8_total_anchors, m_inputWidth);
    AnchorMatrixConstruct(m_stride16_anchors, YOLOV8N_OD_STRIDE_16, m_stride16_total_anchors, m_inputWidth);
    AnchorMatrixConstruct(m_stride32_anchors, YOLOV8N_OD_STRIDE_32, m_stride32_total_anchors, m_inputWidth);
}

void PostProcessor::CalBoxXYWH(
    TfLiteTensor* psBoxOutputTensor,
    std::vector<AnchorBox>& vAnchorBoxes,
    int anchorIndex,
    int stride,
    int totalAnchors,
    Detection& det
)
{
    float scaleBox;
    int zeroPointBox;
    int anchors;
    int boxDataSize;
    float XYWHResult[4];
    
    int8_t* tensorOutputBox = psBoxOutputTensor->data.int8;
    scaleBox = ((TfLiteAffineQuantization *)(psBoxOutputTensor->quantization.params))->scale->data[0];
    zeroPointBox = ((TfLiteAffineQuantization *)(psBoxOutputTensor->quantization.params))->zero_point->data[0];

    anchors = psBoxOutputTensor->dims->data[1];
    boxDataSize = psBoxOutputTensor->dims->data[2];

    if (anchors != totalAnchors)
    {
        return;
    }

    if (boxDataSize != 64)
    {
        return;
    }

    tensorOutputBox = tensorOutputBox + (anchorIndex * boxDataSize);
    
    for (int k = 0; k < 4; k++)
    {
        std::vector<float> XYWHSoftmaxTemp(16);
        float XYWHSoftmaxResult = 0.0f;

        for (int i = 0; i < 16; i++)
        {
            XYWHSoftmaxTemp[i] = scaleBox * (static_cast<float>(tensorOutputBox[k*16 + i]) - zeroPointBox);
        }

        arm::app::math::MathUtils::SoftmaxF32(XYWHSoftmaxTemp);
        for (int i = 0; i < 16; i++)
        {
            XYWHSoftmaxResult = XYWHSoftmaxResult + XYWHSoftmaxTemp[i] * i;
        }
        XYWHResult[k] = XYWHSoftmaxResult;
    }

    /* dist2bbox */
    float x1 = vAnchorBoxes[anchorIndex].w - XYWHResult[0];
    float y1 = vAnchorBoxes[anchorIndex].h - XYWHResult[1];
    float x2 = vAnchorBoxes[anchorIndex].w + XYWHResult[2];
    float y2 = vAnchorBoxes[anchorIndex].h + XYWHResult[3];
    
    float cx = (x1 + x2) / 2.0f;
    float cy = (y1 + y2) / 2.0f;
    float w = x2 - x1;
    float h = y2 - y1;

    XYWHResult[0] = cx * stride;
    XYWHResult[1] = cy * stride;
    XYWHResult[2] = w * stride;
    XYWHResult[3] = h * stride;

    det.x = XYWHResult[0] - (0.5f * XYWHResult[2]);
    det.y = XYWHResult[1] - (0.5f * XYWHResult[3]);
    det.w = XYWHResult[2];
    det.h = XYWHResult[3];
}

static float Calculate1DOverlap(float x1Center, float width1, float x2Center, float width2)
{
    float left_1 = x1Center - width1 / 2.0f;
    float left_2 = x2Center - width2 / 2.0f;
    float leftest = left_1 > left_2 ? left_1 : left_2;

    float right_1 = x1Center + width1 / 2.0f;
    float right_2 = x2Center + width2 / 2.0f;
    float rightest = right_1 < right_2 ? right_1 : right_2;

    return rightest - leftest;
}

static float CalculateBoxIntersect(Detection& box1, Detection& box2)
{
    // Compute center and dimensions to check intersection
    float box1_cx = box1.x + box1.w / 2.0f;
    float box2_cx = box2.x + box2.w / 2.0f;
    float width = Calculate1DOverlap(box1_cx, box1.w, box2_cx, box2.w);
    if (width < 0.0f) {
        return 0.0f;
    }
    float box1_cy = box1.y + box1.h / 2.0f;
    float box2_cy = box2.y + box2.h / 2.0f;
    float height = Calculate1DOverlap(box1_cy, box1.h, box2_cy, box2.h);
    if (height < 0.0f) {
        return 0.0f;
    }

    return width * height;
}

static float CalculateBoxUnion(Detection& box1, Detection& box2)
{
    float boxes_intersection = CalculateBoxIntersect(box1, box2);
    float boxes_union = box1.w * box1.h + box2.w * box2.h - boxes_intersection;
    return boxes_union;
}

static float CalculateBoxIOU(Detection& box1, Detection& box2)
{
    float boxes_intersection = CalculateBoxIntersect(box1, box2);
    if (boxes_intersection == 0.0f) {
        return 0.0f;
    }

    float boxes_union = CalculateBoxUnion(box1, box2);
    if (boxes_union == 0.0f) {
        return 0.0f;
    }

    return boxes_intersection / boxes_union;
}

void PostProcessor::CalculateNMS(std::forward_list<Detection>& detections, float iouThreshold)
{
    int idxClass = 0;
    auto CompareProbs = [idxClass](Detection& prob1, Detection& prob2) {
        return prob1.prob[idxClass] > prob2.prob[idxClass];
    };

    for (idxClass = 0; idxClass < YOLOV8N_OD_CLASS; ++idxClass) {
        detections.sort(CompareProbs);

        for (auto it = detections.begin(); it != detections.end(); ++it) {
            if (it->prob[idxClass] == 0.0f) continue;
            for (auto itc = std::next(it, 1); itc != detections.end(); ++itc) {
                if (itc->prob[idxClass] == 0.0f) {
                    continue;
                }
                if (CalculateBoxIOU(*it, *itc) > iouThreshold) {
                    itc->prob[idxClass] = 0.0f;
                }
            }
        }
    }
}

void PostProcessor::CalDetectionBox(
    TfLiteTensor* psConfidenceOutputTensor,
    TfLiteTensor* psBoxOutputTensor,
    std::vector<AnchorBox>& vAnchorBoxes,
    int stride,
    int totalAnchors,
    float threshold,
    std::forward_list<Detection>& detections
)
{
    float scaleConf;
    int zeroPointConf;
    size_t tensorSizeConf;
    float maxScore = 0.0f;
    int maxConf;
    int cls = 0;
    int8_t* tensorOutputConf = psConfidenceOutputTensor->data.int8;

    scaleConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->scale->data[0];
    zeroPointConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->zero_point->data[0];
    tensorSizeConf = psConfidenceOutputTensor->dims->data[1] * psConfidenceOutputTensor->dims->data[2];

    if ((tensorSizeConf / YOLOV8N_OD_CLASS) != static_cast<size_t>(totalAnchors))
    {
        return;
    }

    for (int i = 0; i < totalAnchors; i++)
    {
        maxScore = 0.0f;
        cls = 0;
        maxConf = -128;

        for (int j = 0; j < YOLOV8N_OD_CLASS; j++)
        {
            int confTensorData = tensorOutputConf[(i * YOLOV8N_OD_CLASS) + j];
            if (confTensorData > maxConf)
            {
                maxConf = confTensorData;
                cls = j;
            }
        }

        maxScore = arm::app::math::MathUtils::SigmoidF32(scaleConf * (static_cast<float>(maxConf - zeroPointConf)));

        if (maxScore >= threshold)
        {
            Detection det;
            det.cls = cls;
            det.score = maxScore;

            for (int j = 0; j < YOLOV8N_OD_CLASS; j++) {
                float score = arm::app::math::MathUtils::SigmoidF32(scaleConf * (static_cast<float>(tensorOutputConf[(i * YOLOV8N_OD_CLASS) + j] - zeroPointConf)));
                det.prob.emplace_back(score);
            }

            CalBoxXYWH(psBoxOutputTensor, vAnchorBoxes, i, stride, totalAnchors, det);
            detections.emplace_front(det);
        }
    }
}

void PostProcessor::GetNetworkBoxes(arm::app::Model* model, std::forward_list<Detection>& detections, float threshold)
{
    TfLiteTensor* psConfidenceTensor;
    TfLiteTensor* psBoxTensor;
    
    psConfidenceTensor = model->GetOutputTensor(YOLOV8N_OD_STRIDE8_CONFIDENCE_TENSOR_INDEX);
    psBoxTensor = model->GetOutputTensor(YOLOV8N_OD_STRIDE8_BOX_TENSOR_INDEX);
    CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride8_anchors, YOLOV8N_OD_STRIDE_8, m_stride8_total_anchors, threshold, detections); 

    psConfidenceTensor = model->GetOutputTensor(YOLOV8N_OD_STRIDE16_CONFIDENCE_TENSOR_INDEX);
    psBoxTensor = model->GetOutputTensor(YOLOV8N_OD_STRIDE16_BOX_TENSOR_INDEX);
    CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride16_anchors, YOLOV8N_OD_STRIDE_16, m_stride16_total_anchors, threshold, detections); 

    psConfidenceTensor = model->GetOutputTensor(YOLOV8N_OD_STRIDE32_CONFIDENCE_TENSOR_INDEX);
    psBoxTensor = model->GetOutputTensor(YOLOV8N_OD_STRIDE32_BOX_TENSOR_INDEX);
    CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride32_anchors, YOLOV8N_OD_STRIDE_32, m_stride32_total_anchors, threshold, detections); 
}

void PostProcessor::Process(
    arm::app::Model* model,
    float threshold,
    Detection* results,
    size_t maxResults,
    size_t& resultCount
)
{
    std::forward_list<Detection> sDetections;
    GetNetworkBoxes(model, sDetections, threshold);
    CalculateNMS(sDetections, 0.45f);

    resultCount = 0;
    for (auto it = sDetections.begin(); it != sDetections.end(); ++it)
    {
        if (it->prob[it->cls] > 0.0f)
        {
            if (resultCount < maxResults)
            {
                // Clip box boundaries to 192x192 space
                float x = std::min(std::max(it->x, 0.0f), static_cast<float>(m_inputWidth - 1));
                float y = std::min(std::max(it->y, 0.0f), static_cast<float>(m_inputHeight - 1));
                float w = std::min(std::max(it->w, 0.0f), static_cast<float>(m_inputWidth - 1));
                float h = std::min(std::max(it->h, 0.0f), static_cast<float>(m_inputHeight - 1));

                results[resultCount++] = {x, y, w, h, it->prob[it->cls], it->cls, {}};
            }
            else
            {
                break;
            }
        }
    }
}

} /* namespace model */
} /* namespace app */
} /* namespace arm */
