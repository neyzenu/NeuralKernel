/*
 * kcompanion.c - AI-driven System Companion Kernel Module
 *
 * This module monitors system metrics, detects anomalies,
 * and provides suggestions to improve system performance.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/swap.h>
#include <linux/proc_fs.h>
#include <linux/kprobes.h>
#include <linux/timekeeping.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/workqueue.h>
#include <linux/sched/signal.h>
#include <linux/string.h>
#include <linux/math64.h>  /* For kernel math functions */

MODULE_LICENSE("GPL");
MODULE_AUTHOR("NUY");
MODULE_DESCRIPTION("AI-driven system companion kernel module");
MODULE_VERSION("0.1");

#define DEVICE_NAME "kcompanion"
#define CLASS_NAME "kcomp"
#define MAX_SUGGESTIONS 10
#define MAX_SUGGESTION_LENGTH 256
#define HISTORY_SIZE 60  // Store 60 data points for each metric
#define ANOMALY_THRESHOLD 2.0  // Standard deviations for anomaly detection
#define SAMPLING_INTERVAL (HZ * 5)  // 5 seconds interval

/* Device variables */
static int major_number;
static struct class *kcompanion_class = NULL;
static struct device *kcompanion_device = NULL;
static struct cdev kcompanion_cdev;

/* Data structures for metrics */
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
    ktime_t timestamp;
};

/* History storage */
static struct system_metrics metrics_history[HISTORY_SIZE];
static int history_index = 0;
static bool history_full = false;

/* Statistics for anomaly detection - using integer arithmetic for kernel space */
struct metric_stats {
    long mean;          /* Fixed point: value * 1000 */
    long std_dev;       /* Fixed point: value * 1000 */
};

static struct metric_stats cpu_user_stats;
static struct metric_stats cpu_system_stats;
static struct metric_stats mem_free_stats;
static struct metric_stats disk_io_stats;

/* Suggestions storage */
static char suggestions[MAX_SUGGESTIONS][MAX_SUGGESTION_LENGTH];
static int suggestion_count = 0;
static DEFINE_MUTEX(suggestions_mutex);

/* Work queue for periodic data collection */
static struct delayed_work metrics_work;
static struct workqueue_struct *metrics_wq;

/* Kprobe for system call tracking */
static struct kprobe kp = {
    .symbol_name = "sys_enter_read",  // Using a common syscall as example
};

static unsigned long syscall_count = 0;
static DEFINE_SPINLOCK(syscall_lock);

/* Kernel math helpers - use integer arithmetic with fixed-point representation */
static long calculate_mean(unsigned long *data, int size)
{
    int i;
    unsigned long long sum = 0;  /* Use unsigned long long to avoid overflow */
    
    for (i = 0; i < size; i++) {
        sum += data[i];
    }
    
    /* Return mean * 1000 as fixed point */
    return size > 0 ? div_u64(sum * 1000ULL, size) : 0;
}

static long calculate_std_dev(unsigned long *data, int size, long mean)
{
    int i;
    unsigned long long sum_squared_diff = 0;
    long diff;
    
    for (i = 0; i < size; i++) {
        /* Calculate (value*1000 - mean) */
        diff = (long)(data[i] * 1000ULL) - mean;
        /* Square the difference and accumulate */
        sum_squared_diff += (unsigned long long)(diff) * (unsigned long long)(diff);
    }
    
    /* Return sqrt(sum_squared_diff / size) * 1000 */
    if (size <= 1)
        return 0;
        
    return int_sqrt(div_u64(sum_squared_diff, size));
}

/* Helper to get absolute value */
static long long abs_val(long long val)
{
    return val < 0 ? -val : val;
}

/* Add a new suggestion */
static void add_suggestion(const char *suggestion)
{
    mutex_lock(&suggestions_mutex);
    
    if (suggestion_count < MAX_SUGGESTIONS) {
        strncpy(suggestions[suggestion_count], suggestion, MAX_SUGGESTION_LENGTH - 1);
        suggestions[suggestion_count][MAX_SUGGESTION_LENGTH - 1] = '\0';
        suggestion_count++;
        pr_info("kcompanion: New suggestion added: %s\n", suggestion);
    } else {
        /* Shift suggestions to make room for new one */
        int i;
        for (i = 0; i < MAX_SUGGESTIONS - 1; i++) {
            strncpy(suggestions[i], suggestions[i + 1], MAX_SUGGESTION_LENGTH);
        }
        strncpy(suggestions[MAX_SUGGESTIONS - 1], suggestion, MAX_SUGGESTION_LENGTH - 1);
        suggestions[MAX_SUGGESTIONS - 1][MAX_SUGGESTION_LENGTH - 1] = '\0';
        pr_info("kcompanion: New suggestion added (replaced oldest): %s\n", suggestion);
    }
    
    mutex_unlock(&suggestions_mutex);
}

/* Get current system metrics */
static void collect_system_metrics(struct system_metrics *metrics)
{
    struct sysinfo i;
    
    si_meminfo(&i);
    
    /* Set timestamp */
    metrics->timestamp = ktime_get();
    
    /* Get CPU metrics from /proc/stat in a real implementation */
    /* This is simplified for demonstration */
    metrics->cpu_user = jiffies % 100;    /* Placeholder */
    metrics->cpu_system = (jiffies / 100) % 50;  /* Placeholder */
    metrics->cpu_idle = 100 - metrics->cpu_user - metrics->cpu_system;
    
    /* Memory metrics */
    metrics->mem_total = i.totalram * i.mem_unit;
    metrics->mem_free = i.freeram * i.mem_unit;
    metrics->mem_available = si_mem_available() * PAGE_SIZE;
    
    /* Disk I/O metrics would come from /proc/diskstats in a real implementation */
    metrics->disk_reads = jiffies % 1000;  /* Placeholder */
    metrics->disk_writes = (jiffies / 100) % 500;  /* Placeholder */
    
    /* Get syscall count with atomics */
    spin_lock(&syscall_lock);
    metrics->syscall_count = syscall_count;
    syscall_count = 0;  /* Reset counter */
    spin_unlock(&syscall_lock);
}

/* Update statistics for anomaly detection */
static void update_statistics(void)
{
    int i;
    int count = history_full ? HISTORY_SIZE : history_index;
    unsigned long cpu_user_data[HISTORY_SIZE];
    unsigned long cpu_system_data[HISTORY_SIZE];
    unsigned long mem_free_data[HISTORY_SIZE];
    unsigned long disk_io_data[HISTORY_SIZE];
    
    /* Extract data into separate arrays for processing */
    for (i = 0; i < count; i++) {
        cpu_user_data[i] = metrics_history[i].cpu_user;
        cpu_system_data[i] = metrics_history[i].cpu_system;
        mem_free_data[i] = metrics_history[i].mem_free;
        disk_io_data[i] = metrics_history[i].disk_reads + metrics_history[i].disk_writes;
    }
    
    /* Calculate statistics using fixed point arithmetic */
    cpu_user_stats.mean = calculate_mean(cpu_user_data, count);
    cpu_user_stats.std_dev = calculate_std_dev(cpu_user_data, count, cpu_user_stats.mean);
    
    cpu_system_stats.mean = calculate_mean(cpu_system_data, count);
    cpu_system_stats.std_dev = calculate_std_dev(cpu_system_data, count, cpu_system_stats.mean);
    
    mem_free_stats.mean = calculate_mean(mem_free_data, count);
    mem_free_stats.std_dev = calculate_std_dev(mem_free_data, count, mem_free_stats.mean);
    
    disk_io_stats.mean = calculate_mean(disk_io_data, count);
    disk_io_stats.std_dev = calculate_std_dev(disk_io_data, count, disk_io_stats.std_dev);
}

/* Detect anomalies and generate suggestions */
static void detect_anomalies(struct system_metrics *current_metrics)
{
    long cpu_user_z, cpu_system_z, mem_free_z, disk_io_z;
    unsigned long disk_io_current;
    long current_value, threshold;
    
    /* Need enough history for meaningful anomaly detection */
    if (!history_full && history_index < 10)
        return;
    
    disk_io_current = current_metrics->disk_reads + current_metrics->disk_writes;
    
    /* Calculate z-scores (how many standard deviations from the mean) */
    /* Using fixed-point arithmetic (values * 1000) */
    
    /* CPU user z-score */
    current_value = (long)(current_metrics->cpu_user * 1000);
    cpu_user_z = abs_val(current_value - cpu_user_stats.mean);
    cpu_user_z = cpu_user_stats.std_dev > 0 ? 
                 div64_ul(cpu_user_z * 1000, cpu_user_stats.std_dev) : 0;
    
    /* CPU system z-score */
    current_value = (long)(current_metrics->cpu_system * 1000);
    cpu_system_z = abs_val(current_value - cpu_system_stats.mean);
    cpu_system_z = cpu_system_stats.std_dev > 0 ? 
                  div64_ul(cpu_system_z * 1000, cpu_system_stats.std_dev) : 0;
    
    /* Memory free z-score */
    current_value = (long)(current_metrics->mem_free * 1000);
    mem_free_z = abs_val(current_value - mem_free_stats.mean);
    mem_free_z = mem_free_stats.std_dev > 0 ? 
                div64_ul(mem_free_z * 1000, mem_free_stats.std_dev) : 0;
    
    /* Disk I/O z-score */
    current_value = (long)(disk_io_current * 1000);
    disk_io_z = abs_val(current_value - disk_io_stats.mean);
    disk_io_z = disk_io_stats.std_dev > 0 ? 
               div64_ul(disk_io_z * 1000, disk_io_stats.std_dev) : 0;
    
    /* Threshold is 2.0 * 1000 in fixed point */
    threshold = 2000;
    
    /* Check for anomalies and generate suggestions */
    if (cpu_user_z > threshold && 
        (current_metrics->cpu_user * 1000) > cpu_user_stats.mean) {
        add_suggestion("High user CPU usage detected. Consider checking for runaway processes.");
    }
    
    if (cpu_system_z > threshold && 
        (current_metrics->cpu_system * 1000) > cpu_system_stats.mean) {
        add_suggestion("High system CPU usage detected. The system might be under heavy I/O or network load.");
    }
    
    if (mem_free_z > threshold && 
        (current_metrics->mem_free * 1000) < mem_free_stats.mean) {
        add_suggestion("Memory pressure detected. Consider closing unused applications.");
    }
    
    if (disk_io_z > threshold && 
        (disk_io_current * 1000) > disk_io_stats.mean) {
        add_suggestion("Unusual disk I/O activity detected. Check for disk-intensive operations.");
    }
    
    /* Additional heuristic rules */
    if (current_metrics->mem_free < current_metrics->mem_total / 10) {
        add_suggestion("Low memory condition: Less than 10% of memory is available.");
    }
    
    if (current_metrics->cpu_idle < 10) {
        add_suggestion("CPU is heavily loaded: Less than 10% idle time.");
    }
    
    if (current_metrics->syscall_count > 10000) {
        add_suggestion("High system call rate detected. Check for busy applications.");
    }
}

/* Handler for metrics collection */
static void metrics_collection_handler(struct work_struct *work)
{
    struct system_metrics current_metrics;
    
    /* Collect current metrics */
    collect_system_metrics(&current_metrics);
    
    /* Store in history */
    metrics_history[history_index] = current_metrics;
    history_index = (history_index + 1) % HISTORY_SIZE;
    if (history_index == 0)
        history_full = true;
    
    /* Update statistics */
    update_statistics();
    
    /* Detect anomalies */
    detect_anomalies(&current_metrics);
    
    /* Schedule next collection */
    queue_delayed_work(metrics_wq, &metrics_work, SAMPLING_INTERVAL);
}

/* Kprobe handler to count syscalls */
static int handler_pre(struct kprobe *p, struct pt_regs *regs)
{
    spin_lock(&syscall_lock);
    syscall_count++;
    spin_unlock(&syscall_lock);
    return 0;
}

/* Device file operations */
static int kcompanion_open(struct inode *inode, struct file *file)
{
    return 0;
}

static int kcompanion_release(struct inode *inode, struct file *file)
{
    return 0;
}

static ssize_t kcompanion_read(struct file *file, char __user *buf,
                              size_t len, loff_t *offset)
{
    char temp_buf[MAX_SUGGESTION_LENGTH];
    size_t temp_len;
    int i, total_len = 0;
    
    if (*offset > 0)
        return 0;  /* EOF */
    
    mutex_lock(&suggestions_mutex);
    
    /* Prepare output buffer with all suggestions */
    if (suggestion_count == 0) {
        strncpy(temp_buf, "No suggestions available.\n", MAX_SUGGESTION_LENGTH - 1);
        temp_buf[MAX_SUGGESTION_LENGTH - 1] = '\0';
        temp_len = strlen(temp_buf);
        
        if (copy_to_user(buf, temp_buf, temp_len)) {
            mutex_unlock(&suggestions_mutex);
            return -EFAULT;
        }
        
        *offset += temp_len;
        total_len = temp_len;
    } else {
        for (i = 0; i < suggestion_count && total_len < len; i++) {
            snprintf(temp_buf, MAX_SUGGESTION_LENGTH, "%d: %s\n", i + 1, suggestions[i]);
            temp_len = strlen(temp_buf);
            
            if (total_len + temp_len > len)
                break;
            
            if (copy_to_user(buf + total_len, temp_buf, temp_len)) {
                mutex_unlock(&suggestions_mutex);
                return -EFAULT;
            }
            
            total_len += temp_len;
        }
        
        *offset += total_len;
    }
    
    mutex_unlock(&suggestions_mutex);
    return total_len;
}

static ssize_t kcompanion_write(struct file *file, const char __user *buf,
                               size_t len, loff_t *offset)
{
    char feedback[MAX_SUGGESTION_LENGTH];
    size_t feedback_len = len < MAX_SUGGESTION_LENGTH - 1 ? len : MAX_SUGGESTION_LENGTH - 1;
    
    if (copy_from_user(feedback, buf, feedback_len)) {
        return -EFAULT;
    }
    
    feedback[feedback_len] = '\0';
    
    /* Handle user feedback */
    if (strncmp(feedback, "good:", 5) == 0) {
        pr_info("kcompanion: Positive feedback received: %s\n", feedback + 5);
        /* In a real implementation, we would use this to adjust ML parameters */
    } else if (strncmp(feedback, "bad:", 4) == 0) {
        pr_info("kcompanion: Negative feedback received: %s\n", feedback + 4);
        /* In a real implementation, we would use this to adjust ML parameters */
    } else {
        /* Treat as a custom suggestion */
        add_suggestion(feedback);
    }
    
    *offset += len;
    return len;
}

/* IOCTL commands */
#define KCOMPANION_CLEAR_SUGGESTIONS _IO('k', 1)
#define KCOMPANION_GET_METRICS _IOR('k', 2, struct system_metrics)
#define KCOMPANION_RESET_STATS _IO('k', 3)

static long kcompanion_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct system_metrics current_metrics;
    
    switch (cmd) {
        case KCOMPANION_CLEAR_SUGGESTIONS:
            mutex_lock(&suggestions_mutex);
            suggestion_count = 0;
            mutex_unlock(&suggestions_mutex);
            pr_info("kcompanion: Suggestions cleared\n");
            return 0;
            
        case KCOMPANION_GET_METRICS:
            collect_system_metrics(&current_metrics);
            if (copy_to_user((struct system_metrics *)arg, &current_metrics, 
                            sizeof(struct system_metrics))) {
                return -EFAULT;
            }
            return 0;
            
        case KCOMPANION_RESET_STATS:
            history_index = 0;
            history_full = false;
            pr_info("kcompanion: Statistics reset\n");
            return 0;
            
        default:
            return -ENOTTY; /* Unknown command */
    }
}

/* File operations structure */
static const struct file_operations kcompanion_fops = {
    .owner = THIS_MODULE,
    .open = kcompanion_open,
    .release = kcompanion_release,
    .read = kcompanion_read,
    .write = kcompanion_write,
    .unlocked_ioctl = kcompanion_ioctl,
};

/* Module initialization */
static int __init kcompanion_init(void)
{
    int ret;
    
    /* Register character device */
    major_number = register_chrdev(0, DEVICE_NAME, &kcompanion_fops);
    if (major_number < 0) {
        pr_err("kcompanion: Failed to register a major number\n");
        return major_number;
    }
    
    /* Register device class - API changed in kernel 6.8 */
    kcompanion_class = class_create(CLASS_NAME);
    if (IS_ERR(kcompanion_class)) {
        unregister_chrdev(major_number, DEVICE_NAME);
        pr_err("kcompanion: Failed to register device class\n");
        return PTR_ERR(kcompanion_class);
    }
    
    /* Create device */
    kcompanion_device = device_create(kcompanion_class, NULL, 
                                    MKDEV(major_number, 0), 
                                    NULL, DEVICE_NAME);
    if (IS_ERR(kcompanion_device)) {
        class_destroy(kcompanion_class);
        unregister_chrdev(major_number, DEVICE_NAME);
        pr_err("kcompanion: Failed to create the device\n");
        return PTR_ERR(kcompanion_device);
    }
    
    /* Initialize workqueue */
    metrics_wq = create_singlethread_workqueue("kcompanion_metrics");
    if (!metrics_wq) {
        device_destroy(kcompanion_class, MKDEV(major_number, 0));
        class_destroy(kcompanion_class);
        unregister_chrdev(major_number, DEVICE_NAME);
        pr_err("kcompanion: Failed to create workqueue\n");
        return -ENOMEM;
    }
    
    /* Initialize kprobe for syscall tracking */
    kp.pre_handler = handler_pre;
    ret = register_kprobe(&kp);
    if (ret < 0) {
        pr_warn("kcompanion: Kprobe registration failed, syscall tracking disabled: %d\n", ret);
        /* Continue without syscall tracking */
    }
    
    /* Schedule first metrics collection */
    INIT_DELAYED_WORK(&metrics_work, metrics_collection_handler);
    queue_delayed_work(metrics_wq, &metrics_work, SAMPLING_INTERVAL);
    
    /* Add initial suggestion */
    add_suggestion("kCompanion AI is now active and monitoring your system.");
    
    pr_info("kcompanion: Module loaded successfully\n");
    pr_info("kcompanion: Created device /dev/%s\n", DEVICE_NAME);
    
    return 0;
}

/* Module cleanup */
static void __exit kcompanion_exit(void)
{
    /* Cancel and flush workqueue */
    cancel_delayed_work_sync(&metrics_work);
    flush_workqueue(metrics_wq);
    destroy_workqueue(metrics_wq);
    
    /* Unregister kprobe */
    unregister_kprobe(&kp);
    
    /* Clean up device */
    device_destroy(kcompanion_class, MKDEV(major_number, 0));
    class_destroy(kcompanion_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    
    pr_info("kcompanion: Module unloaded successfully\n");
}

module_init(kcompanion_init);
module_exit(kcompanion_exit);

/* The system in kcompanion uses simple statistical methods, not true AI/ML:
 *
 * 1. Data Collection (lines ~180-207):
 *    - collect_system_metrics(): Gathers CPU, memory, disk I/O metrics
 *    - The metrics_collection_handler() work queue handler regularly collects data
 *
 * 2. Statistical Analysis (lines ~125-160):
 *    - calculate_mean() and calculate_std_dev(): Calculate basic statistical metrics
 *    - update_statistics(): Updates statistical models based on historical data
 *    - This is simple descriptive statistics, not machine learning
 *
 * 3. Anomaly Detection (lines ~210-255):
 *    - detect_anomalies(): Uses z-score statistical method to identify outliers
 *    - This is basic statistical threshold-based detection, not AI
 *    - Uses fixed thresholds (ANOMALY_THRESHOLD) rather than learned parameters
 *
 * 4. Feedback Collection (lines ~310-330):
 *    - kcompanion_write(): Collects user feedback with "good:" and "bad:" prefixes
 *    - This feedback is logged but not actually used to improve the system
 *    - No actual learning from feedback is implemented
 */
