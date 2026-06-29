/**************************************************************************//**
 * @file     InferenceModel.hpp
 * @version  V1.00
 * @brief    Model class wrapper header
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __INFERENCE_MODEL_HPP__
#define __INFERENCE_MODEL_HPP__

#include "Model.hpp"

namespace arm
{
namespace app
{

class InferenceModel : public Model
{
public:
    /* Indices for expected model input shape */
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

protected:
    /** @brief   Gets the reference to op resolver interface class. */
    const tflite::MicroOpResolver &GetOpResolver() override;

    /** @brief   Adds operations to the op resolver instance. */
    bool EnlistOperations() override;

private:
    /* Maximum number of individual operations that can be enlisted for CPU/NPU. */
    static constexpr int ms_maxOpCnt = 8;

    /* A mutable op resolver instance. */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif // __INFERENCE_MODEL_HPP__
