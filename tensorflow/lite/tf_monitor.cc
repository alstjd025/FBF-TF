#include "tensorflow/lite/tf_monitor.h"

namespace tflite{

LiteSysMonitor::LiteSysMonitor(){
  std::cout << "Init system monitoring" << "\n";
}

LiteSysMonitor::LiteSysMonitor(float* cpu_util, float* gpu_util){
  cpu_util_ = cpu_util;
  gpu_util_ = gpu_util;
  std::cout << "System monitoring started" << "\n";
  CPU_daemon = std::thread(&LiteSysMonitor::GetCPUUtilization, this);
  GPU_daemon = std::thread(&LiteSysMonitor::GetGPUUtilization, this);
  CPU_daemon.detach();
  GPU_daemon.detach();
}

LiteSysMonitor::~LiteSysMonitor(){
  // must terminate CPU_daemon & GPU_daemon here.
  std::cout << "System monitoring terminated" << "\n";
}

struct cpuusage LiteSysMonitor::GetCPUusageFromCpustat(struct cpustat s) {
  struct cpuusage r;
  strncpy(r.name, s.name, sizeof(r.name));
  r.name[sizeof(r.name) - 1] = '\0';
  r.idletime = s.idle + s.iowait;
  r.workingtime = s.user + s.nice + s.system + s.irq + s.softirq;
  return r;
}

float LiteSysMonitor::CpuUsageGetDiff(struct cpuusage now, struct cpuusage prev) {
  // the number of ticks that passed by since the last measurement.
  const unsigned long long workingtime = now.workingtime - prev.workingtime;
  const unsigned long long alltime = workingtime + (now.idletime - prev.idletime);
  // they are divided by themselves - so the unit does not matter.
  // printf("CPU Usage: %.0Lf%%\n", (long double)workingtime / alltime * 100.0L);
  return (float)(workingtime / alltime * 100.0L);
}

// Simply parses /proc/stat.
void LiteSysMonitor::GetCPUUtilization() {
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
        *cpu_util_ = CpuUsageGetDiff(now, prev);
        prev = now;
        break;
      }
    }
    fclose(f);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

// For jetson platforms only.
// Uses tegrastats
void LiteSysMonitor::GetGPUUtilization(){
  std::string data, cmd;
  cmd = "tegrastats --interval 10";
  FILE* stream;
  const int max_buffer = 512;
  char buffer[max_buffer];
  stream = popen(cmd.c_str(), "r");
  if (stream) {
    while (!feof(stream)) {
      if (fgets(buffer, max_buffer, stream) != NULL) {
        data.append(buffer);
      }
      int delimiter = data.find("GR3D_FREQ");
      std::string front = data.substr(delimiter+5);
      delimiter = front.find('%');
      front = front.substr(0, delimiter);
      delimiter = front.find(' ');
      front = front.substr(delimiter+1);
      // std::cout << "GPU : " << front << "\n";
      *gpu_util_ = std::stoi(front);
      // std::cout << "GPU Usage: "<< *gpu_util_ << "% \n"; 
      data.clear();
    }
    pclose(stream);
  }
}

} // namespace tflite
