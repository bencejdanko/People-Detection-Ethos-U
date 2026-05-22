/**************************************************************************//**
 * @file     PostProcessor.cpp
 * @version  V1.00
 * @brief    C++ post-processing source for peak detection and NMS
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "PostProcessor.hpp"

namespace arm
{
namespace app
{
namespace model
{

PostProcessor::PostProcessor(int inputWidth, int inputHeight, int gridWidth, int gridHeight)
    : m_inputWidth(inputWidth),
      m_inputHeight(inputHeight),
      m_gridWidth(gridWidth),
      m_gridHeight(gridHeight)
{
}

void PostProcessor::Process(const int8_t* outputData,
                                 float threshold,
                                 float minDistance,
                                 float scale,
                                 int32_t zeroPoint,
                                 Detection* results,
                                 size_t maxResults,
                                 size_t& resultCount)
{
    resultCount = 0;
    const float minDistanceSq = minDistance * minDistance;

    for (int y = 0; y < m_gridHeight; ++y)
    {
        for (int x = 0; x < m_gridWidth; ++x)
        {
            // The output tensor shape is [1, gridHeight, gridWidth, num_classes]
            // We assume num_classes = 2, where:
            //   Index 0 = Background
            //   Index 1 = Person (target class)
            int tensorIdx = (y * m_gridWidth + x) * 2 + 1;
            int8_t quantizedVal = outputData[tensorIdx];
            
            // Dequantize: float = (quantized - zero_point) * scale
            float score = (static_cast<float>(quantizedVal) - static_cast<float>(zeroPoint)) * scale;

            if (score >= threshold)
            {
                Detection candidate;
                candidate.grid_x = x;
                candidate.grid_y = y;
                candidate.score = score;
                // Coordinates in image space
                candidate.x = (static_cast<float>(x) + 0.5f) * (static_cast<float>(m_inputWidth) / static_cast<float>(m_gridWidth));
                candidate.y = (static_cast<float>(y) + 0.5f) * (static_cast<float>(m_inputHeight) / static_cast<float>(m_gridHeight));

                bool merged = false;
                for (size_t i = 0; i < resultCount; ++i)
                {
                    float dx = static_cast<float>(candidate.grid_x - results[i].grid_x);
                    float dy = static_cast<float>(candidate.grid_y - results[i].grid_y);

                    if ((dx * dx + dy * dy) < minDistanceSq)
                    {
                        if (candidate.score > results[i].score)
                        {
                            results[i] = candidate;
                        }
                        merged = true;
                        break;
                    }
                }

                if (!merged && resultCount < maxResults)
                {
                    results[resultCount++] = candidate;
                }
            }
        }
    }
}

} /* namespace model */
} /* namespace app */
} /* namespace arm */
