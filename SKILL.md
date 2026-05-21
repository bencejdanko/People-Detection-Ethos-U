# NuMaker-X-M55M1D Memory Management and Toolchain Integration

This document describes the memory layout, allocation strategy, and early-boot constraints for the Nuvoton NuMaker-X-M55M1D edge AI firmware. It serves as a guide for managing tightly coupled memories, on-chip SRAM banks, and external HyperRAM when integrating large machine learning models, OpenMV frame buffers, and LwIP network stacks.

---

## 1. Physical Memory Architecture

The Nuvoton M55M1H2LJAE microcontroller features a multi-tier memory layout optimized for high-performance neural network execution (via the Ethos-U55 NPU) and low-latency interrupt handling:

| Memory Region | Base Address | Size | Access Type | Primary Use Case |
|---|---|---|---|---|
| ITCM | 0x00000000 | 64 KB | Secure / Single-cycle | Core instruction vector and critical interrupt routines (e.g. OpenMV core libraries). |
| DTCM | 0x20000000 | 128 KB | Secure / Single-cycle | Stack, heap, and time-critical global variables (e.g., FreeRTOS kernels). |
| FLASH (APROM) | 0x00100000 | 2 MB | Secure | Read-only application code, weights, and static lookup tables. |
| SRAM01 | 0x20100000 | 1 MB | Secure / Cached | General-purpose on-chip SRAM; used for model activation buffers. |
| SRAM2 | 0x20200000 | 320 KB | Secure / Non-cached | Secondary on-chip SRAM; used for OpenMV display buffers and LwIP pools. |
| SPIM0 (HyperRAM) | 0x82000000 | 32 MB | Secure / External | Gigantic external RAM; used for mounting model files from the SD card. |

---

## 2. Crucial Constraints on Early-Boot Initialization

A common point of failure in Cortex-M55 systems is the hardware power-on-reset state of the memory controllers.

* Power State on Reset: Only the ITCM, DTCM, and internal Flash are powered on and accessible immediately upon reset. All other SRAM banks (SRAM0, SRAM1, SRAM2) and external interfaces (SPIM0 HyperRAM) have their clocks disabled by the System Clock Controller to conserve energy.
* The Role of __main__: The C-runtime initialization library (`__main__`) runs immediately after the assembly reset handler and before entering `main()`. It processes the linker-generated scatter load table to copy initialized variables (`+RW`) from Flash to RAM and to zero-initialize uninitialized variables (`+ZI`).
* The Bus Lockup Hazard: If any general-purpose variable (`.ANY (+RW +ZI)`) is placed in a disabled memory bank (such as SRAM01 or external HyperRAM), `__main__` will attempt to write to that address during early initialization. This immediately triggers a hardware Bus Fault (Memory Transfer Fault) before `BoardInit()` or `SystemInit()` can enable the clocks. This traps the CPU in an infinite lockup loop and prevents the debugger (pyOCD) from halting or writing to DTCM.

---

## 3. Memory Allocation Strategy

To accommodate large neural networks alongside the LwIP TCP/IP stack without causing linker overflows or early-boot lockups, the following separation of concerns is implemented in `M55M1.scatter`:

### A. Tightly Coupled Data Memory (DTCM)
DTCM is reserved strictly for components that must be present and fully accessible immediately upon CPU power-up:
* System Stack (`ARM_LIB_STACK`): Shrunk from 40 KB to 12 KB (`0x3000`). This is used for the pre-scheduler system execution and ISRs.
* System Heap (`ARM_LIB_HEAP`): Shrunk from 64 KB to 2 KB (`0x800`). Since TensorFlow, OpenMV, and LwIP use dedicated static arrays, the C standard library heap is almost completely bypassed.
* Standard Variables (`.ANY (+RW +ZI)`): All standard initialized and zero-initialized system variables remain in DTCM.

### B. Shared On-Chip RAM 2 (SRAM2)
SRAM2 is configured as an uninitialized execution region (`UNINIT`). This prevents `__main__` from touching it during boot:
* Large LwIP Memory Buffers: Large LwIP structures (such as `ram_heap` [16 KB], `memp_memory_PBUF_POOL_base` [24 KB], and `memp_memory_emac_rx_base` [EMAC RX pool]) are explicitly directed into SRAM2 using wildcards (e.g. `*mem.o` and `*memp.o`). 
* Runtime Initialization: LwIP is self-initializing. Its internal APIs (`mem_init()` and `memp_init()`) clear and format these buffers at runtime when `tcpip_init()` is called, making the startup `UNINIT` status safe and memory-efficient.
* OpenMV VRAM: OpenMV frame buffers (`fb_array`, `jpeg_array`, and `frame_buf1`) are directed to SRAM2. Since OpenMV is initialized programmatically inside `main()`, they do not require early-boot zeroing.

### C. Shared On-Chip RAM 0, 1 (SRAM01)
SRAM01 is designated for the TensorFlow Lite Micro model execution space:
* Model Activation Arena: `tensorArena` [1 MB] is statically defined inside `main.cpp` within the `.bss.NoInit.activation_buf_sram` section and mapped to SRAM01 under the `UNINIT` tag. The model wrapper initializes this space programmatically when launching the Ethos-U55 NPU.

---

## 4. Recovering from Debugger Memory Faults

When a bus fault locks up the Cortex-M55 core, pyOCD will report:
`Memory transfer fault @ 0x20007c78-0x20007fff [__main__]`

To unfreeze the debugger and load a corrected binary:
1. Hold down the physical RESET button on the NuMaker-X-M55M1D board.
2. Trigger the `CMSIS Load` build task (or `pyocd load` command) in VS Code.
3. Release the RESET button immediately when the first lines of the flash discovery process appear in the terminal.
4. The debugger will catch the CPU vector fetch, halt the execution, and flash the corrected image successfully.
