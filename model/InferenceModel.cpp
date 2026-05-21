/**************************************************************************//**
 * @file     InferenceModel.cpp
 * @version  V1.00
 * @brief    Model class wrapper implementation
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "InferenceModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::InferenceModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::InferenceModel::EnlistOperations()
{
#if defined(ARM_NPU)
    // Add Ethos-U custom operator for hardware acceleration
    if (kTfLiteOk == this->m_opResolver.AddEthosU())
    {
        info("Added %s support to op resolver\n", tflite::GetString_ETHOSU());
    }
    else
    {
        printf_err("Failed to add Arm NPU support to op resolver.\n");
        return false;
    }
#endif /* ARM_NPU */

    if (this->m_opResolver.AddConv2D() != kTfLiteOk) {
        printf_err("Failed to add Conv2D to op resolver\n");
        return false;
    }
    
    if (this->m_opResolver.AddDepthwiseConv2D() != kTfLiteOk) {
        printf_err("Failed to add DepthwiseConv2D to op resolver\n");
        return false;
    }

    if (this->m_opResolver.AddSoftmax() != kTfLiteOk) {
        printf_err("Failed to add Softmax to op resolver\n");
        return false;
    }

    if (this->m_opResolver.AddReshape() != kTfLiteOk) {
        printf_err("Failed to add Reshape to op resolver\n");
        return false;
    }

    return true;
}
