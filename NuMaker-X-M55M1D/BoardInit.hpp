/**************************************************************************//**
 * @file     BoardInit.hpp
 * @version  V1.00
 * @brief    Target board initialization header file (Ethernet + NPU support)
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __BOARD_INIT_HPP__
#define __BOARD_INIT_HPP__

/**
  * @brief Initiate the hardware resources of board
  * @return 0: Success, <0: Fail
  * @details Initiate clock, UART, NPU, HyperRAM, SD Card, and Ethernet MAC
  * \hideinitializer
  */
int BoardInit(void);

#endif // __BOARD_INIT_HPP__
