#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <iostream>
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

struct cpuusage cpuusage_from_cpustat(struct cpustat s) {
    struct cpuusage r;
    strncpy(r.name, s.name, sizeof(r.name));
    r.name[sizeof(r.name) - 1] = '\0';
    r.idletime = s.idle + s.iowait;
    r.workingtime = s.user + s.nice + s.system + s.irq + s.softirq;
    return r;
}

void cpuusage_show_diff(struct cpuusage now, struct cpuusage prev) {
    // the number of ticks that passed by since the last measurement
    const unsigned long long workingtime = now.workingtime - prev.workingtime;
    const unsigned long long alltime = workingtime + (now.idletime - prev.idletime);
    // they are divided by themselves - so the unit does not matter.
    printf("Usage: %.0Lf%%\n", (long double)workingtime / alltime * 100.0L);
}

void GetCPUUtilization(float& util) {
    struct cpuusage prev = {0};
    //
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
        FILE *f = fmemopen(buffer, readed, "r");
        // Uch, so much borign typing.
        struct cpustat c = {0};
        while (fscanf(f, "%19s %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", c.name, &c.user, &c.nice,
                  &c.system, &c.idle, &c.iowait, &c.irq, &c.softirq, &c.steal, &c.guest,
                  &c.guest_nice) == 11) {
            // Just an example for first cpu core.
            if (strcmp(c.name, "cpu") == 0) {
                struct cpuusage now = cpuusage_from_cpustat(c);
                cpuusage_show_diff(now, prev);
                prev = now;
                break;
            }
        }
        fclose(f);
        //
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // sleep(0.8);
    }
}

void GetGPUUtilization(float& util){
std::string data, cmd;
  cmd = "tegrastats --interval 10";
  // cmd = "tegrastats";
  char separator = '[';
  FILE * stream;
  const int max_buffer = 512;
  char buffer[max_buffer];
  std::vector<int> cpu_util;
  int num_cpu = std::thread::hardware_concurrency();
  std::cout << "HW has " << num_cpu << " cores" << "\n";
  stream = popen(cmd.c_str(), "r");
  if (stream) {
    while (!feof(stream)) {
      if (fgets(buffer, max_buffer, stream) != NULL) {
        data.append(buffer);
      }
      // std::cout << "========================" << "\n";
      // std::cout << "data : " << data << "\n";
      // std::cout << "========================" << "\n";
      int delimiter = data.find("GR3D_FREQ");
      std::string front = data.substr(delimiter+5);
      int delimiter_ = front.find('%');
      front = front.substr(0, delimiter_);
      std::cout << front << "\n"; 
      // for(int i=0; i<num_cpu; ++i){
      //   int cpu_idx = front.find('%');
      //   cpu_util.push_back(std::stoi(front.substr(0, cpu_idx)));
      //   front = front.substr(cpu_idx, cpu_idx+1);
      // }
      // for(auto cpu_u : cpu_util)
      //   std::cout << cpu_u << " ";
      // std::cout << "\n";
      // cpu_util.clear();
      data.clear();
    }
    pclose(stream);
  }
}


// std::vector<size_t> get_cpu_times() {
//     std::ifstream proc_stat("/proc/stat");
//     proc_stat.ignore(5, ' '); // Skip the 'cpu' prefix.
//     std::vector<size_t> times;
//     for (size_t time; proc_stat >> time; times.push_back(time));
//     return times;
// }

// bool get_cpu_times(size_t &idle_time, size_t &total_time) {
//     const std::vector<size_t> cpu_times = get_cpu_times();
//     if (cpu_times.size() < 4)
//         return false;
//     idle_time = cpu_times[3];
//     total_time = std::accumulate(cpu_times.begin(), cpu_times.end(), 0);
//     return true;
// }

// int main(int, char *[]) {
//     size_t previous_idle_time=0, previous_total_time=0;
//     for (size_t idle_time, total_time; get_cpu_times(idle_time, total_time); sleep(1)) {
//         const float idle_time_delta = idle_time - previous_idle_time;
//         const float total_time_delta = total_time - previous_total_time;
//         const float utilization = 100.0 * (1.0 - idle_time_delta / total_time_delta);
//         std::cout << utilization << '%' << std::endl;
//         previous_idle_time = idle_time;
//         previous_total_time = total_time;
//     }
// }

// void startMonitoring(std::string& stat){
 
// }