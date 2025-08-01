/*
 * kcompanion_client.h - Header file for the kCompanion client
 */

#ifndef KCOMPANION_CLIENT_H
#define KCOMPANION_CLIENT_H

#define DEVICE_PATH "/dev/kcompanion"
#define MAX_BUFFER_SIZE 4096

/* IOCTL commands (must match those in the kernel module) */
#define KCOMPANION_CLEAR_SUGGESTIONS _IO('k', 1)
#define KCOMPANION_GET_METRICS _IOR('k', 2, struct system_metrics)
#define KCOMPANION_RESET_STATS _IO('k', 3)

/* Must match the definition in the kernel module */
struct system_metrics {
    /* CPU metrics */
    unsigned long cpu_user;
    unsigned long cpu_system;
    unsigned long cpu_idle;
    
    /* Memory metrics */
    unsigned long mem_total;
    unsigned long mem_free;
    unsigned long mem_available;
    
    /* Disk I/O metrics */
    unsigned long disk_reads;
    unsigned long disk_writes;
    
    /* System call count */
    unsigned long syscall_count;
    
    /* Timestamp */
    long long timestamp;
};

/* Function prototypes */
void print_help(void);
void get_suggestions(int fd);
void show_metrics(int fd);
void clear_suggestions(int fd);
void reset_stats(int fd);
void send_feedback(int fd, const char *message);
void add_suggestion(int fd, const char *suggestion);

#endif /* KCOMPANION_CLIENT_H */
