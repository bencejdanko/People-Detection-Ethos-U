/**************************************************************************//**
 * @file     BoardInit.cpp
 * @version  V1.00
 * @brief    Target board initialization implementation (Ethernet, NPU, HyperRAM)
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <cstdio>
// Avoid ISO C++17 'register' storage class specifier errors in old Nuvoton SDK headers
#define register
#include "NuMicro.h"
#include "log_macros.h"

#include "ethosu_npu_init.h"
#include "hyperram_code.h"
#include "board_config.h"

#define DESIGN_NAME "NuMaker-X-M55M1D"
#define HYPERRAM_SPIM_PORT SPIM0        // Standard port for NuMaker-M55M1 boards

static void SDCard_PinConfig(void)
{
    /* Set multi-function pins for SDH0 */
    SET_SD0_nCD_PD13();
    SET_SD0_CLK_PE6();
    SET_SD0_CMD_PE7();
    SET_SD0_DAT0_PE2();
    SET_SD0_DAT1_PE3();
    SET_SD0_DAT2_PE4();
    SET_SD0_DAT3_PE5();

    /* Enable internal pull-up on PD13 (nCD) to prevent floating state when no card is inserted */
    GPIO_SetPullCtl(PD, BIT13, GPIO_PUSEL_PULL_UP);
}

static void Ethernet_PinConfig(void)
{
    /* Set multi-function pins for EMAC0 RMII */
    SET_EMAC0_RMII_MDC_PE8();
    SET_EMAC0_RMII_MDIO_PE9();
    SET_EMAC0_RMII_TXD0_PE10();
    SET_EMAC0_RMII_TXD1_PE11();
    SET_EMAC0_RMII_TXEN_PE12();
    SET_EMAC0_RMII_REFCLK_PC8();
    SET_EMAC0_RMII_RXD0_PC7();
    SET_EMAC0_RMII_RXD1_PC6();
    SET_EMAC0_RMII_CRSDV_PA7();
    SET_EMAC0_RMII_RXERR_PA6();

    /* Fast slew control on PE.10, PE.11, PE.12 for Ethernet RMII */
    GPIO_SetSlewCtl(PE, (BIT10 | BIT11 | BIT12), GPIO_SLEWCTL_FAST0);

    /* PE.13 Set high to enable Ethernet PHY */
    GPIO_SetMode(PE, BIT13, GPIO_MODE_OUTPUT);
    PE13 = 1;
}

static void SYS_Init(void)
{
    /*---------------------------------------------------------------------------------------------------------*/
    /* Init System Clock                                                                                       */
    /*---------------------------------------------------------------------------------------------------------*/
    
    /* Enable Internal RC 12MHz clock */
    CLK_EnableXtalRC(CLK_SRCCTL_HIRCEN_Msk);

    /* Waiting for Internal RC clock ready */
    CLK_WaitClockReady(CLK_STATUS_HIRCSTB_Msk);

    /* Enable HXT clock */
    CLK_EnableXtalRC(CLK_SRCCTL_HXTEN_Msk);

    /* Waiting for HXT clock ready */
    CLK_WaitClockReady(CLK_STATUS_HXTSTB_Msk);

    /* Switch SCLK clock source to APLL0 and Enable APLL0 220MHz clock */
    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0, CLK_APLLCTL_APLLSRC_HXT, FREQ_220MHZ);

    /* Enable APLL1 clock */
    CLK_EnableAPLL(CLK_APLLCTL_APLLSRC_HXT, FREQ_220MHZ, CLK_APLL1_SELECT);

    /* Update System Core Clock */
    SystemCoreClockUpdate();

    /* Enable GPIO module clocks */
    CLK_EnableModuleClock(GPIOA_MODULE);
    CLK_EnableModuleClock(GPIOB_MODULE);
    CLK_EnableModuleClock(GPIOC_MODULE);
    CLK_EnableModuleClock(GPIOD_MODULE);
    CLK_EnableModuleClock(GPIOE_MODULE);
    CLK_EnableModuleClock(GPIOF_MODULE);
    CLK_EnableModuleClock(GPIOG_MODULE);
    CLK_EnableModuleClock(GPIOH_MODULE);
    CLK_EnableModuleClock(GPIOI_MODULE);
    CLK_EnableModuleClock(GPIOJ_MODULE);

    /* Enable FMC0 module clock to keep FMC clock when CPU is idle but NPU is running */
    CLK_EnableModuleClock(FMC0_MODULE);

    /* Enable NPU module clock */
    CLK_EnableModuleClock(NPU0_MODULE);

#if USE_CCAP_CAMERA
    /* Enable CCAP0 module clock */
    CLK_EnableModuleClock(CCAP0_MODULE);
#endif /* USE_CCAP_CAMERA */

    /* Enable SDH0 module clock source as HCLK0 and SDH0 module clock divider as 4 */
    CLK_EnableModuleClock(SDH0_MODULE);
    CLK_SetModuleClock(SDH0_MODULE, CLK_SDHSEL_SDH0SEL_HCLK0, CLK_SDHDIV_SDH0DIV(4));

    /* Enable Ethernet MAC (EMAC0) module clock */
    CLK_EnableModuleClock(EMAC0_MODULE);
    SYS_ResetModule(SYS_EMAC0RST);

    /* Select UART module clock source and clock divider */
    SetDebugUartCLK();

    /*---------------------------------------------------------------------------------------------------------*/
    /* Init I/O Multi-function                                                                                 */
    /*---------------------------------------------------------------------------------------------------------*/
    
    /* Set multi-function pins for Debug UART RXD and TXD */
    SetDebugUartMFP();

    /* Set up HyperRAM pins */
    HyperRAM_PinConfig(HYPERRAM_SPIM_PORT);

    /* Set up SD Card pins */
    SDCard_PinConfig();

    /* Set up Ethernet MAC RMII pins */
    Ethernet_PinConfig();
}

int BoardInit(void)
{
    /* Unlock protected registers */
    SYS_UnlockReg();

    /* Initialize System Clocks and Pin Multiplexing */
    SYS_Init();

    /* Initialize Debug UART (115200-8N1) for serial logging */
    InitDebugUart();

    /* Lock protected registers */
    SYS_LockReg();

    /* Initialize HyperRAM memory extension */
    HyperRAM_Init(HYPERRAM_SPIM_PORT);
    
    /* Enter direct-mapped mode to allow seamless memory-mapped model reading */
    SPIM_HYPER_EnterDirectMapMode(HYPERRAM_SPIM_PORT);

    /* Open SD Card Disk for loading external files */
    SDH_Open_Disk(SDH0, CardDetect_From_GPIO);

    LOG_INFO("Hardware peripherals initialized.");

#if defined(ARM_NPU)
    int state;
    /* Initialize Arm Ethos-U55 NPU coprocessor */
    LOG_INFO("Initializing Arm Ethos-U55 NPU...");
    if (0 != (state = arm_ethosu_npu_init()))
    {
        LOG_ERROR("Failed to initialize NPU (status: %d)", state);
        return state;
    }
#endif /* ARM_NPU */

    LOG_INFO("Target system: %s", DESIGN_NAME);
    return 0;
}
