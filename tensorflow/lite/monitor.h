#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <iostream>
#include <fcntl.h>
#include <vector>

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
    LiteSysMonitor(float& cpu_util, float& gpu_util);
    ~LiteSysMonitor();

    void GetCPUUtilization(float& util);
    void GetGPUUtilization(float& util);

    struct cpuusage GetCPUusageFromCpustat(struct cpustat s);
    float CpuUsageGetDiff(struct cpuusage now, struct cpuusage prev);

  private:
    std::thread CPU_daemon;
    std::thread GPU_daemon;
  // FILE gpu_stream;
  // FILE cpu_stream;

};

