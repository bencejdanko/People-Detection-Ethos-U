/**************************************************************************//**
 * @file     PostProcessor.cpp
 * @version  V1.00
 * @brief    C++ post-processing source for peak detection and NMS
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "PostProcessor.hpp"
#include <algorithm>
#include <cmath>

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
                                 std::vector<Detection>& results)
{
    results.clear();
    std::vector<Detection> candidates;

    /* 1. Extract peaks above threshold */
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
                
                candidates.push_back(candidate);
            }
        }
    }

    /* 2. Sort candidates by score descending */
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    /* 3. Non-Maximum Suppression (NMS) in grid space */
    for (const auto& candidate : candidates)
    {
        bool tooClose = false;
        for (const auto& accepted : results)
        {
            float dx = static_cast<float>(candidate.grid_x - accepted.grid_x);
            float dy = static_cast<float>(candidate.grid_y - accepted.grid_y);
            float distance = std::sqrt(dx * dx + dy * dy);

            if (distance < minDistance)
            {
                tooClose = true;
                break;
            }
        }

        if (!tooClose)
        {
            results.push_back(candidate);
        }
    }
}

} /* namespace model */
} /* namespace app */
} /* namespace arm */
