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
#include "opencv2/opencv.hpp"
#include <functional>
#include "thread"
#include "future"
#include "tensorflow/lite/util.h"

namespace tflite{
  class TfScheduler{
    public:
      TfScheduler();
      TfScheduler(const char* uds_file_name);

      ~TfScheduler();
    
    private:
    int server_fd;
    std::thread listen_thread;    
  };

}