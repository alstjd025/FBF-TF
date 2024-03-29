#include "tensorflow/lite/tf_monitor.h"
// #define nvidia
#define MONITORING_PERIOD_MS 100 // < 5 is not stable.
#define Experiment


namespace tflite{

LiteSysMonitor::LiteSysMonitor(){
  log_File.open("utilization.txt");
  
  std::cout << "System monitoring started" << "\n";
  CPU_daemon = std::thread(&LiteSysMonitor::GetCPUUtilization, this);
  #ifdef nvidia
    GPU_daemon = std::thread(&LiteSysMonitor::GetGPUUtilization, this);
  #endif
  debugger_daemon = std::thread(&LiteSysMonitor::usage_debugger, this);
  CPU_daemon.detach();
  #ifdef nvidia
    GPU_daemon.detach();
  #endif
  debugger_daemon.detach();
}

LiteSysMonitor::~LiteSysMonitor(){
  // must terminate CPU_daemon & GPU_daemon here.
  std::cout << "System monitoring terminated" << "\n";
}

float LiteSysMonitor::GetCPUUtil(){
  return cpu_util_ratio;
}

float LiteSysMonitor::GetGPUUtil(){
  return gpu_util_ratio;
}

void LiteSysMonitor::usage_debugger(){
  if(log_File.is_open()){
    struct timespec now;
    log_File << "t CPU GPU" << "\n";
    while(1){
      std::this_thread::sleep_for(std::chrono::milliseconds(MONITORING_PERIOD_MS));
      clock_gettime(CLOCK_MONOTONIC, &now);
      log_File << now.tv_sec << "." << now.tv_nsec << " ";
      log_File << cpu_util_ratio << " " << gpu_util_ratio << "\n";
      // logFile << 
    }
  }
}

struct cpuusage LiteSysMonitor::GetCPUusageFromCpustat(struct cpustat s) {
  struct cpuusage r;
  strncpy(r.name, s.name, sizeof(r.name));
  r.name[sizeof(r.name) - 1] = '\0';
  r.idletime = s.idle + s.iowait;
  r.workingtime = s.user + s.nice + s.system + s.irq + s.softirq;
  return r;
}



long double LiteSysMonitor::CpuUsageGetDiff(struct cpuusage now, struct cpuusage prev) {
  // the number of ticks that passed by since the last measurement.
  const unsigned long long workingtime = now.workingtime - prev.workingtime;
  const unsigned long long alltime = workingtime + (now.idletime - prev.idletime);
  // they are divided by themselves - so the unit does not matter.
  // printf("CPU Usage: %.0Lf%%\n", (long double)workingtime / alltime * 100.0L);
  return (long double)workingtime / alltime * 100.0L;
}

// Simply parses /proc/stat.
void LiteSysMonitor::GetCPUUtilization() {
  #ifdef Experiment
  while(1){
    std::ifstream cpu_util;
    cpu_util.open("cpu_util");
    if (!cpu_util.is_open()) {
      std::cout << "CPU util file open error" << "\n";
      return;
    }
    int ratio = 0;
    cpu_util >> ratio;
    cpu_util_ratio = float(ratio);
    cpu_util.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(MONITORING_PERIOD_MS));
  }
  #endif
  #ifndef Experiment
  struct cpuusage prev = {0};
  const int stat = open("/proc/stat", O_RDONLY);
  assert(stat != -1);
  fcntl(stat, F_SETFL, O_NONBLOCK);
  while (1) {
    // let's read everything in one call so it's nicely synced.
    int r = lseek(stat, SEEK_SET, 0);
    assert(r != -1);
    char buffer[10001];
    const ssize_t readed = read(stat, buffer, sizeof(buffer) - 1);
    assert(readed != -1);
    buffer[readed] = '\0';
    // Read the values from the readed buffer/
    FILE* f = fmemopen(buffer, readed, "r");
    // Uch, so much borign typing.
    struct cpustat c = {0};
    while (fscanf(f, "%19s %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", c.name, &c.user, &c.nice,
      &c.system, &c.idle, &c.iowait, &c.irq, &c.softirq, &c.steal, &c.guest,
      &c.guest_nice) == 11) {
      // Just an example for first cpu core.
      if (strcmp(c.name, "cpu") == 0) {
        struct cpuusage now = GetCPUusageFromCpustat(c);
        cpu_util_ratio = CpuUsageGetDiff(now, prev);
        // std::cout << "CPU Usage: " << (int)cpu_util_ratio << "% \n";
        prev = now;
        break;
      }
    }
    fclose(f);
    std::this_thread::sleep_for(std::chrono::milliseconds(MONITORING_PERIOD_MS));
  }
  #endif // !Experiment
}

void LiteSysMonitor::GetGPUUtilization() {
  #ifdef Experiment
  while(1){
    std::ifstream gpu_util;
    gpu_util.open(("gpu_util"));
    if (!gpu_util.is_open()) {
      std::cout << "GPU util file open error" << "\n";
      return;
    }
    int ratio = 0;
    gpu_util >> ratio;
    gpu_util_ratio = float(ratio);
    gpu_util.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(MONITORING_PERIOD_MS));
  }
  #endif
  #ifndef Experiment
  struct cpuusage prev = {0};
  const int stat = open("/sys/devices/gpu.0/load", O_RDONLY);
  assert(stat != -1);
  fcntl(stat, F_SETFL, O_NONBLOCK);
  while (1) {
    // let's read everything in one call so it's nicely synced.
    int r = lseek(stat, SEEK_SET, 0);
    assert(r != -1);
    char buffer[8];
    const ssize_t readed = read(stat, buffer, sizeof(buffer) - 1);
    assert(readed != -1);
    buffer[readed] = '\0';
    // Read the values from the readed buffer/
    FILE* f = fmemopen(buffer, readed, "r");
    // Uch, so much borign typing.
    int percentage = 0;
    while (fscanf(f, "%llu", &percentage)) {
      gpu_util_ratio = percentage / 10;
      // std::cout << "GPU Usage: " << gpu_util_ratio << "% \n"; 
      break;
    }
    fclose(f);
    std::this_thread::sleep_for(std::chrono::milliseconds(MONITORING_PERIOD_MS));
  }
  #endif
}

} // namespace tflite
