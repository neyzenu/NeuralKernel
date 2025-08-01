#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/ioctl.h>
#include "kcompanion_client.h"

#define MAX_BUFFER_SIZE 256

void print_help(void)
{
    printf("kCompanion Client - Interact with the kCompanion kernel module\n\n");
    printf("Usage:\n");
    printf("  get        - Get current suggestions\n");
    printf("  metrics    - Show current system metrics\n");
    printf("  clear      - Clear all suggestions\n");
    printf("  reset      - Reset statistics collection\n");
    printf("  feedback <message> - Send feedback to the AI system\n");
    printf("  suggest <message>  - Add a custom suggestion\n");
    printf("  help       - Show this help message\n");
    printf("  exit/quit  - Exit the program\n\n");
    printf("Note: The kCompanion AI daemon should be running for advanced features.\n");
}

void get_suggestions(int fd)
{
    char buffer[MAX_BUFFER_SIZE];
    ssize_t bytes_read;
    
    /* Read suggestions from the device */
    bytes_read = read(fd, buffer, sizeof(buffer) - 1);
    if (bytes_read < 0) {
        perror("Failed to read suggestions");
        return;
    }
    
    buffer[bytes_read] = '\0';
    printf("--- Current Suggestions ---\n%s\n", buffer);
}

void show_metrics(int fd)
{
    struct system_metrics metrics;
    
    if (ioctl(fd, KCOMPANION_GET_METRICS, &metrics) < 0) {
        perror("Failed to get system metrics");
        return;
    }
    
    printf("--- Current System Metrics ---\n");
    printf("CPU User:   %lu%%\n", metrics.cpu_user);
    printf("CPU System: %lu%%\n", metrics.cpu_system);
    printf("CPU Idle:   %lu%%\n", metrics.cpu_idle);
    printf("Memory Total:     %lu KB\n", metrics.mem_total / 1024);
    printf("Memory Free:      %lu KB\n", metrics.mem_free / 1024);
    printf("Memory Available: %lu KB\n", metrics.mem_available / 1024);
    printf("Disk Reads:  %lu operations\n", metrics.disk_reads);
    printf("Disk Writes: %lu operations\n", metrics.disk_writes);
    printf("System Calls: %lu calls\n", metrics.syscall_count);
    printf("\n");
}

void clear_suggestions(int fd)
{
    if (ioctl(fd, KCOMPANION_CLEAR_SUGGESTIONS) < 0) {
        perror("Failed to clear suggestions");
        return;
    }
    printf("All suggestions have been cleared.\n");
}

void reset_stats(int fd)
{
    if (ioctl(fd, KCOMPANION_RESET_STATS) < 0) {
        perror("Failed to reset statistics");
        return;
    }
    printf("Statistics have been reset. The system will begin learning from scratch.\n");
}

void send_feedback(int fd, const char *message)
{
    char buffer[MAX_BUFFER_SIZE];
    snprintf(buffer, sizeof(buffer), "good:%s", message);
    
    if (write(fd, buffer, strlen(buffer)) < 0) {
        perror("Failed to send feedback");
        return;
    }
    printf("Feedback sent successfully.\n");
}

void add_suggestion(int fd, const char *suggestion)
{
    if (write(fd, suggestion, strlen(suggestion)) < 0) {
        perror("Failed to add suggestion");
        return;
    }
    printf("Suggestion added successfully.\n");
}

int main(void)
{
    int fd;
    char cmd[MAX_BUFFER_SIZE];
    char *arg;
    
    /* Open the device */
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return EXIT_FAILURE;
    }
    
    printf("kCompanion Client\n");
    printf("Type 'help' for available commands\n");
    
    while (1) {
        printf("> ");
        if (!fgets(cmd, sizeof(cmd), stdin)) {
            break;
        }
        
        /* Remove newline */
        cmd[strcspn(cmd, "\n")] = 0;
        
        /* Extract the first word (command) */
        arg = cmd;
        while (*arg && !isspace(*arg)) {
            arg++;
        }
        if (*arg) {
            *arg++ = '\0';  /* null-terminate command and point arg to the rest */
            while (*arg && isspace(*arg)) {
                arg++;       /* skip leading spaces in argument */
            }
        }
        
        if (strcmp(cmd, "exit") == 0 || strcmp(cmd, "quit") == 0) {
            break;
        } else if (strcmp(cmd, "help") == 0) {
            print_help();
        } else if (strcmp(cmd, "get") == 0) {
            get_suggestions(fd);
        } else if (strcmp(cmd, "metrics") == 0) {
            show_metrics(fd);
        } else if (strcmp(cmd, "clear") == 0) {
            clear_suggestions(fd);
        } else if (strcmp(cmd, "reset") == 0) {
            reset_stats(fd);
        } else if (strcmp(cmd, "feedback") == 0) {
            if (*arg) {
                send_feedback(fd, arg);
            } else {
                printf("Error: 'feedback' command requires a message\n");
            }
        } else if (strcmp(cmd, "suggest") == 0) {
            if (*arg) {
                add_suggestion(fd, arg);
            } else {
                printf("Error: 'suggest' command requires a message\n");
            }
        } else {
            printf("Unknown command. Type 'help' for available commands.\n");
        }
    }
    
    close(fd);
    return EXIT_SUCCESS;
}