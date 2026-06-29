#ifndef __WIFI_PUSH_HPP__
#define __WIFI_PUSH_HPP__

#include <stdint.h>

namespace WifiPush {
    // Configure Wi-Fi credentials and target server details
    void Configure(const char* ssid, const char* pass, const char* host, uint16_t port, const char* path);

    // Initialize the ESP8266 serial interface, hardware control pins, and start the background FreeRTOS task
    void Start(void);

    // Enqueue a new count value to be sent by the background task
    void PushCount(int count);

    // Check if the board is connected to the Wi-Fi Access Point
    bool IsConnected(void);
}

#endif // __WIFI_PUSH_HPP__
