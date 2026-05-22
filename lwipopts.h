#ifndef LWIP_LWIPOPTS_H
#define LWIP_LWIPOPTS_H

#include "board_config.h"

#define NO_SYS                          0

/* Core locking */
#define LWIP_TCPIP_CORE_LOCKING         1

#define SYS_LIGHTWEIGHT_PROT            1

/* Memory options */
#define MEM_ALIGNMENT                   4
#define MEM_SIZE                        (16 * 1024)

#define MEMP_NUM_PBUF                   16
#define MEMP_NUM_UDP_PCB                4
#define MEMP_NUM_TCP_PCB                4
#define MEMP_NUM_TCP_PCB_LISTEN         4
#define MEMP_NUM_TCP_SEG                16
#define MEMP_NUM_SYS_TIMEOUT            8
#define MEMP_NUM_NETBUF                 8
#define MEMP_NUM_NETCONN                8
#define MEMP_NUM_TCPIP_MSG_API          8
#define MEMP_NUM_TCPIP_MSG_INPKT        8

/* Pbuf options */
#define PBUF_POOL_SIZE                  16
#define PBUF_POOL_BUFSIZE               1536

/* TCP/IP thread options */
#define TCPIP_THREAD_NAME               "tcpip"
#define TCPIP_THREAD_STACKSIZE          1024
#define TCPIP_THREAD_PRIO               (tskIDLE_PRIORITY + 5UL)
#define TCPIP_MBOX_SIZE                 8

#define RX_THREAD_STACKSIZE             1024
#define RX_THREAD_PRIO                  (tskIDLE_PRIORITY + 5UL)

/* Internal memory pool sizes */
#define DEFAULT_RAW_RECVMBOX_SIZE       8
#define DEFAULT_UDP_RECVMBOX_SIZE       8
#define DEFAULT_TCP_RECVMBOX_SIZE       8
#define DEFAULT_ACCEPTMBOX_SIZE         8

/* TCP/IP API options */
#define LWIP_NETCONN                    1
#define LWIP_SOCKET                     1
#define LWIP_STATS                      0
#define LWIP_TIMERS                     1
#define LWIP_TIMERS_CUSTOM              0

/* Protocol options */
#define LWIP_ETHERNET                   1
#define LWIP_ARP                        1
#define LWIP_IP                         1
#define LWIP_RAW                        1
#define LWIP_UDP                        1
#define LWIP_TCP                        1
#define LWIP_DHCP                       LWIP_DHCP_ENABLE

#define TCP_MSS                         1460
#define TCP_WND                         (4 * TCP_MSS)
#define TCP_SND_BUF                     (4 * TCP_MSS)
#define TCP_SND_QUEUELEN                16

/* Checksum options */
#define CHECKSUM_GEN_IP                 1
#define CHECKSUM_GEN_UDP                1
#define CHECKSUM_GEN_TCP                1
#define CHECKSUM_CHECK_IP               1
#define CHECKSUM_CHECK_UDP              1
#define CHECKSUM_CHECK_TCP              1

#undef SYS_TIMEOUT

#endif /* LWIP_LWIPOPTS_H */
