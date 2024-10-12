#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <vector>
#include <atomic>

namespace tflite{

struct cpuusage {
  char name[20];
  // Absolute values since last reboot.
  unsigned long long idletime;
  unsigned long long workingtime;
};

struct cpustat {
  char name[20];
  unsigned long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
};

class LiteSysMonitor{
  public:
    LiteSysMonitor();
    ~LiteSysMonitor();

    void GetCPUUtilization();
    void GetGPUUtilization();
    void usage_debugger();
    void GlobalResourceMonitor();

    int GetRecoveryFD();

    float GetGPUUtil();
    float GetCPUUtil();


    struct cpuusage GetCPUusageFromCpustat(struct cpustat s);
    long double CpuUsageGetDiff(struct cpuusage now, struct cpuusage prev);

    std::ofstream log_File; 
  private:
    // file discriptor for recovery alert.
    int recovery_fds[2];
    int recovery_fd_wr;
    int recovery_fd_rd;

    std::thread CPU_daemon;
    std::thread GPU_daemon;
    std::thread debugger_daemon;
    
    std::atomic<float> cpu_util_ratio; 
    std::atomic<float> gpu_util_ratio; 
};

} // namespace tflite

