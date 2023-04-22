#pragma once
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <vector>
#include <utility>
#include <queue>
#include "condition_variable"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <error.h>
#include "opencv2/opencv.hpp"
#include <functional>
#include "thread"
#include "future"
#include "tensorflow/lite/util.h"

namespace tflite{

  typedef struct runtime_{
    int id;
    RuntimeState state;
    struct sockaddr_un addr;
    float latency[1000];
    int partitioning_plan[1000][3];
  }runtime_;

  class TfScheduler{
    public:
      TfScheduler();
      TfScheduler(const char* uds_file_name);
      void Work();

      int SendPacketToRuntime(tf_packet& tx_p, struct sockaddr_un& runtime_addr);
      
      int ReceivePacketFromRuntime(tf_packet& rx_p, struct sockaddr_un& runtime_addr);
      
      // refresh runtime state in scheduler.
      void RefreshRuntimeState(tf_packet& rx_p);

      void CreatePartitioningPlan(tf_packet& rx_p, tf_packet& tx_p);

      ~TfScheduler();
    
    private:
    int scheduler_fd;
    size_t addr_size;
    struct sockaddr_un scheduler_addr;

    std::vector<runtime_*> runtimes;
    int runtimes_created = 0;
  };

}