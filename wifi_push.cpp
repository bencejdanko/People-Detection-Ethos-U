#include "wifi_push.hpp"
#include "NuMicro.h"
#include "FreeRTOS.h"
#include "task.h"
#include "board_config.h"
#include <stdio.h>
#include <string.h>

#define WIFI_UART       UART8
#define RST_PIN         PD2

namespace WifiPush {
    static char g_ssid[64] = "";
    static char g_pass[64] = "";
    static char g_host[64] = "";
    static uint16_t g_port = 80;
    static char g_path[64] = "/count";

    static volatile int g_latestCount = 0;
    static volatile bool g_newCountAvailable = false;

    void Configure(const char* ssid, const char* pass, const char* host, uint16_t port, const char* path) {
        strncpy(g_ssid, ssid, sizeof(g_ssid) - 1);
        strncpy(g_pass, pass, sizeof(g_pass) - 1);
        strncpy(g_host, host, sizeof(g_host) - 1);
        g_port = port;
        strncpy(g_path, path, sizeof(g_path) - 1);
        
        printf("[WIFI] Configured with SSID: %s, Host: %s, Port: %d, Path: %s\n", 
               g_ssid, g_host, g_port, g_path);
    }

    void PushCount(int count) {
        g_latestCount = count;
        g_newCountAvailable = true;
    }

    static void UART_ResetFIFO(void) {
        WIFI_UART->FIFOSTS |= (UART_FIFOSTS_RXRST_Msk | UART_FIFOSTS_TXRST_Msk);
    }

    static void Wifi_UART_Init(void) {
        SYS_UnlockReg();

        /* Enable UART8 clock */
        CLK_EnableModuleClock(UART8_MODULE);
        /* Select UART8 module clock source as HIRC and UART8 module clock divider as 1 */
        CLK_SetModuleClock(UART8_MODULE, CLK_UARTSEL1_UART8SEL_HIRC, CLK_UARTDIV1_UART8DIV(1));
        /* Reset UART8 module */
        SYS_ResetModule(SYS_UART8RST);

        /* Set multi-function pins for RXD, TXD, CTS and RTS */
        SET_UART8_TXD_PJ0();
        SET_UART8_RXD_PJ1();
        SET_UART8_nCTS_PI14();
        SET_UART8_nRTS_PI15();

        SYS_LockReg();

        /* Open UART8 at 115200 bps */
        UART_Open(WIFI_UART, 115200);
    }

    static void Wifi_Reset_Hardware(void) {
        GPIO_SetMode(PD, BIT2, GPIO_MODE_OUTPUT);

        printf("[WIFI] Hard-resetting ESP8266...\n");
        RST_PIN = 0;
        vTaskDelay(pdMS_TO_TICKS(1000));
        RST_PIN = 1;
        
        // Wait for bootloader to finish starting up
        vTaskDelay(pdMS_TO_TICKS(3000));
        UART_ResetFIFO();
    }

    static bool SendCommand(const char* cmd, const char* expected_resp, uint32_t timeout_ms) {
        // Clear RX FIFO first
        while ((WIFI_UART->FIFOSTS & UART_FIFOSTS_RXEMPTY_Msk) == 0) {
            volatile uint32_t dummy = WIFI_UART->DAT;
        }

        // Send command
        UART_Write(WIFI_UART, (const uint8_t*)cmd, strlen(cmd));

        // Read response
        static char resp_buf[1024];
        size_t idx = 0;
        uint32_t start_tick = xTaskGetTickCount();
        while ((xTaskGetTickCount() - start_tick) < pdMS_TO_TICKS(timeout_ms)) {
            if ((WIFI_UART->FIFOSTS & UART_FIFOSTS_RXEMPTY_Msk) == 0) {
                char c = WIFI_UART->DAT;
                if (idx < sizeof(resp_buf) - 1) {
                    resp_buf[idx++] = c;
                    resp_buf[idx] = '\0';
                }
                if (expected_resp && strstr(resp_buf, expected_resp) != NULL) {
                    return true;
                }
            } else {
                vTaskDelay(pdMS_TO_TICKS(10));
            }
        }
        return false;
    }

    static bool ConnectToAP(void) {
        Wifi_Reset_Hardware();

        if (!SendCommand("AT\r\n", "OK", 2000)) {
            printf("[WIFI] Module not responding to AT commands.\n");
            return false;
        }

        // Disable command echo for cleaner response parsing
        SendCommand("ATE0\r\n", "OK", 1000);

        printf("[WIFI] Setting station mode (CWMODE=1)...\n");
        SendCommand("AT+CWMODE=1\r\n", "OK", 2000);

        char connect_cmd[128];
        snprintf(connect_cmd, sizeof(connect_cmd), "AT+CWJAP=\"%s\",\"%s\"\r\n", g_ssid, g_pass);
        printf("[WIFI] Connecting to Access Point: %s...\n", g_ssid);
        
        if (!SendCommand(connect_cmd, "OK", 15000)) {
            printf("[WIFI] Failed to connect to Access Point.\n");
            return false;
        }
        printf("[WIFI] Connected to Access Point successfully!\n");
        return true;
    }

    static bool SendHTTPPost(int count) {
        char conn_cmd[128];
        snprintf(conn_cmd, sizeof(conn_cmd), "AT+CIPSTART=\"TCP\",\"%s\",%d\r\n", g_host, g_port);
        printf("[WIFI] Opening TCP connection to %s:%d...\n", g_host, g_port);
        
        if (!SendCommand(conn_cmd, "OK", 5000) && !SendCommand("", "ALREADY CONNECTED", 100)) {
            printf("[WIFI] TCP connection failed.\n");
            return false;
        }

        // Format JSON payload
        char payload[128];
        int payload_len = snprintf(payload, sizeof(payload), "{\"count\":%d,\"token\":\"%s\"}", count, SERVER_TOKEN);

        // Format HTTP POST request
        char http_req[512];
        int req_len = snprintf(http_req, sizeof(http_req),
            "POST %s HTTP/1.1\r\n"
            "Host: %s\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n"
            "\r\n"
            "%s",
            g_path, g_host, payload_len, payload);

        char send_cmd[32];
        snprintf(send_cmd, sizeof(send_cmd), "AT+CIPSEND=%d\r\n", req_len);
        
        if (!SendCommand(send_cmd, ">", 2000)) {
            printf("[WIFI] AT+CIPSEND preparation failed.\n");
            SendCommand("AT+CIPCLOSE\r\n", "OK", 1000);
            return false;
        }

        // Transmit request content
        if (!SendCommand(http_req, "SEND OK", 5000)) {
            printf("[WIFI] Data transmission failed.\n");
            SendCommand("AT+CIPCLOSE\r\n", "OK", 1000);
            return false;
        }

        printf("[WIFI] Successfully pushed count: %d\n", count);
        
        // Gracefully close TCP connection
        SendCommand("AT+CIPCLOSE\r\n", "OK", 1000);
        return true;
    }

    static void vWifiPushTask(void* pvParameters) {
        (void)pvParameters;

        Wifi_UART_Init();

        bool wifi_connected = false;
        int last_sent_count = -1;
        uint32_t last_sent_time = 0;

        while (1) {
            if (!wifi_connected) {
                if (strlen(g_ssid) > 0) {
                    wifi_connected = ConnectToAP();
                }
                if (!wifi_connected) {
                    vTaskDelay(pdMS_TO_TICKS(10000));
                    continue;
                }
            }

            uint32_t now = xTaskGetTickCount();
            bool should_push = false;
            int current_count = g_latestCount;

            // Trigger push if count changes or 15-second heartbeat has elapsed
            if (g_newCountAvailable && current_count != last_sent_count) {
                should_push = true;
                g_newCountAvailable = false;
            } else if ((now - last_sent_time) >= pdMS_TO_TICKS(15000)) {
                should_push = true;
            }

            if (should_push) {
                if (SendHTTPPost(current_count)) {
                    last_sent_count = current_count;
                    last_sent_time = xTaskGetTickCount();
                } else {
                    printf("[WIFI] Network transfer failed, reconnecting...\n");
                    wifi_connected = false;
                }
            }

            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    void Start(void) {
        xTaskCreate(vWifiPushTask, "WifiPush", 1024, NULL, tskIDLE_PRIORITY + 1UL, NULL);
    }
}
