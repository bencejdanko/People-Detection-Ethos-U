/**************************************************************************//**
 * @file     PostProcessor.hpp
 * @version  V1.00
 * @brief    C++ post-processing for grid-based detection models
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __POST_PROCESSOR_HPP__
#define __POST_PROCESSOR_HPP__

#include <vector>
#include <cstdint>

namespace arm
{
namespace app
{
namespace model
{

struct Detection
{
    int grid_x;       // X coordinate in output grid space (0 to grid_size - 1)
    int grid_y;       // Y coordinate in output grid space (0 to grid_size - 1)
    float x;          // Scaled coordinate in input image space (0 to input_width)
    float y;          // Scaled coordinate in input image space (0 to input_height)
    float score;      // Confidence score (0.0 to 1.0)
};

class PostProcessor
{
public:
    PostProcessor(int inputWidth, int inputHeight, int gridWidth, int gridHeight);
    ~PostProcessor() = default;

    /**
     * @brief Run post-processing on raw int8 output tensor from NPU
     * @param outputData Pointer to raw int8 output data from model
     * @param threshold Confidence threshold (0.0 to 1.0)
     * @param minDistance Grid-space NMS threshold
     * @param scale Output tensor scale parameter
     * @param zeroPoint Output tensor zero-point parameter
     * @param results Vector to populate with detections
     */
    void Process(const int8_t* outputData,
                 float threshold,
                 float minDistance,
                 float scale,
                 int32_t zeroPoint,
                 std::vector<Detection>& results);

private:
    int m_inputWidth;
    int m_inputHeight;
    int m_gridWidth;
    int m_gridHeight;
};

} /* namespace model */
} /* namespace app */
} /* namespace arm */

#endif // __POST_PROCESSOR_HPP__
