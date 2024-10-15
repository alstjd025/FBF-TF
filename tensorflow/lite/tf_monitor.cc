#include "tensorflow/lite/tf_monitor.h"

// Note: activate if nvidia platform
#define nvidia

// Note: global reosurce monitoring period.
// too short monitoring period might occur overhead.
#define MONITORING_PERIOD_MS 20

// Note: experimental flag for ramdisk writing.
// #define ramdisk_gpu_debug
// #define GPU_UTIL_FILE "/mnt/ramdisk/gpu_util"

// Note: experimental flag to monitor utilization file.
// #define Experiment

// Note: activate monitoring debugger
// #define monitoring_debugger

#define debug_msgs

namespace tflite{

LiteSysMonitor::LiteSysMonitor(){
  log_File.open("utilization.txt");
  if (pipe(recovery_fds) == -1) {
    std::cerr << "Failed to create pipe" << std::endl;
    return;
  }
  recovery_fd_rd = recovery_fds[0];
  recovery_fd_wr = recovery_fds[1];
  if (pipe(recovery_from_sched_fds) == -1) {
    std::cerr << "Failed to create pipe" << std::endl;
    return;
  }
  recovery_fd_rd_sched = recovery_from_sched_fds[0];
  recovery_fd_wr_sched = recovery_from_sched_fds[1];
  std::cout << "pipe read " << recovery_fd_rd << "\n";
  std::cout << "pipe wirte " << recovery_fd_wr << "\n";
  std::cout << "System monitoring started" << "\n";
  CPU_daemon = std::thread(&LiteSysMonitor::GetCPUUtilization, this);
  #ifdef nvidia
    GPU_daemon = std::thread(&LiteSysMonitor::GetGPUUtilization, this);
  #endif
  #ifdef monitoring_debugger
  debugger_daemon = std::thread(&LiteSysMonitor::usage_debugger, this);
  debugger_daemon.detach();
  #endif
  CPU_daemon.detach();
  #ifdef nvidia
    GPU_daemon.detach();
  #endif
  global_monitor_daemon = std::thread(&LiteSysMonitor::GlobalResourceMonitor, this);
  global_monitor_daemon.detach();
}

LiteSysMonitor::~LiteSysMonitor(){
  // must terminate CPU_daemon & GPU_daemon here.
  std::cout << "System monitoring terminated" << "\n";
}

void LiteSysMonitor::GlobalResourceMonitor(){
  float prev_cpu_util = 0;
  float prev_gpu_util = 0;
  bool do_revocery = false;
  float cpu_recovery_threshold = 20;
  float cpu_busy_threshold = 80;
  float gpu_recovery_threshold = 20;
  float gpu_busy_threshold = 80;
  bool recovery_possible = false;
  int activity = 0;
  while(true){
    if(!recovery_possible){
      #ifdef debug_msgs
      std::cout << "select wait" << "\n";
      #endif
      char buffer[4];
      memset(buffer, 0, sizeof(buffer));
      ssize_t bytes_read = read(recovery_fd_rd_sched, buffer, sizeof(buffer));
      #ifdef debug_msgs
      std::cout << "read something" << "\n";
      #endif
      if(cpu_util_ratio < cpu_recovery_threshold && gpu_util_ratio > gpu_busy_threshold
        || gpu_util_ratio < gpu_recovery_threshold && cpu_util_ratio > cpu_busy_threshold){
        #ifdef debug_msgs
        std::cout << "monitor : Do recovery trigger CPU: " << cpu_util_ratio << " GPU: " 
                  << gpu_util_ratio << "\n";
        #endif
        memset(buffer, 1, sizeof(buffer));
        if(write(recovery_fd_wr, buffer, sizeof(buffer)) == -1){
          std::cout << "recovery monitor write(pipe) failed" << "\n";
          return;
        }
      }else{
        std::cout << "monitor : no recovery occurs" << "\n";
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(MONITORING_PERIOD_MS));
  }
}

int LiteSysMonitor::GetRecoveryFDRead(){
  return recovery_fd_rd;
}

int LiteSysMonitor::GetRecoveryFDWrite(){
  return recovery_fd_wr_sched;
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
  struct timespec now;
  #ifdef ramdisk_gpu_debug
    std::ofstream gpu_util_f;
    gpu_util_f.open(GPU_UTIL_FILE, std::ios::out | std::ios::trunc);
    if (!gpu_util_f.is_open()) {
      std::cerr << "Failed to open gpu_util(h)" << std::endl;
      return;
    }
  #endif
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
    clock_gettime(CLOCK_MONOTONIC, &now);
    const ssize_t readed = read(stat, buffer, sizeof(buffer) - 1);
    assert(readed != -1);
    buffer[readed] = '\0';
    // Read the values from the readed buffer/
    FILE* f = fmemopen(buffer, readed, "r");
    int percentage = 0;
    while (fscanf(f, "%llu", &percentage)) {
      gpu_util_ratio = percentage / 10;
      // std::cout << "GPU util: " << gpu_util_ratio << "% \n"; 
      break;
    }
    fclose(f);
    // gpu_util_f << gpu_util_ratio << " " << now.tv_sec << "." << now.tv_nsec /  1000000000.0 << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(MONITORING_PERIOD_MS));
  }
  // gpu_util_f.close();
  #endif
}

} // namespace tflite
