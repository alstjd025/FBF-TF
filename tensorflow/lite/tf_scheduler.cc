#include "tensorflow/lite/tf_scheduler.h"
// #define single_level_motivation

namespace tflite {

TfScheduler::TfScheduler(){};

TfScheduler::TfScheduler(const char* uds_file_name, const char* uds_file_name_sec,
                         std::vector<std::string>& param_file_names) {
  // delete if sock file already exists.
  if (access(uds_file_name, F_OK) == 0) unlink(uds_file_name);

  scheduler_fd = socket(PF_FILE, SOCK_DGRAM, 0);
  if (scheduler_fd == -1) {
    std::cout << "Socket create ERROR"
              << "\n";
    exit(-1);
  }
  addr_size = sizeof(scheduler_addr);

  memset(&scheduler_addr, 0, sizeof(scheduler_addr));
  scheduler_addr.sun_family = AF_UNIX;
  strcpy(scheduler_addr.sun_path, uds_file_name);

  if (bind(scheduler_fd, (struct sockaddr*)&scheduler_addr,
           sizeof(scheduler_addr)) == -1) {
    std::cout << "Socket bind ERROR"
              << "\n";
    exit(-1);
  }

  if (access(uds_file_name_sec, F_OK) == 0) unlink(uds_file_name_sec);

  scheduler_fd_sec = socket(PF_FILE, SOCK_DGRAM, 0);
  if (scheduler_fd_sec == -1) {
    std::cout << "Secondary socket create ERROR"
              << "\n";
    exit(-1);
  }
  addr_size = sizeof(scheduler_addr_sec);

  memset(&scheduler_addr_sec, 0, sizeof(scheduler_addr_sec));
  scheduler_addr_sec.sun_family = AF_UNIX;
  strcpy(scheduler_addr_sec.sun_path, uds_file_name_sec);

  if (bind(scheduler_fd_sec, (struct sockaddr*)&scheduler_addr_sec,
           sizeof(scheduler_addr_sec)) == -1) {
    std::cout << "Secondary socket bind ERROR"
              << "\n";
    exit(-1);
  }

  cpu_util = new float;
  gpu_util = new float;

  OpenPartitioningParams(param_file_names);
  std::cout << "Scheduler initializing done"
            << "\n";
}

int TfScheduler::SendPacketToRuntime(tf_packet& tx_p,
                                     struct sockaddr_un& runtime_addr) {
  int v;
  v = sendto(scheduler_fd, (void*)&tx_p, sizeof(tf_packet), 0,
             (struct sockaddr*)&runtime_addr, sizeof(runtime_addr));
  return v;
}

int TfScheduler::SendPacketToRuntime(tf_runtime_packet& tx_p,
                                     struct sockaddr_un& runtime_addr) {
  int v;
  v = sendto(scheduler_fd, (void*)&tx_p, sizeof(tf_runtime_packet), 0,
             (struct sockaddr*)&runtime_addr, sizeof(runtime_addr));
  return v;
}

int TfScheduler::SendPacketToRuntime(tf_initialization_packet& tx_p,
                                     struct sockaddr_un& runtime_addr) {
  int v;
  v = sendto(scheduler_fd, (void*)&tx_p, sizeof(tf_initialization_packet), 0,
             (struct sockaddr*)&runtime_addr, sizeof(runtime_addr));
  return v;
}

int TfScheduler::SendPacketToRuntimeSecSocket(tf_runtime_packet& tx_p,
                                     struct sockaddr_un& runtime_addr) {
  int v;
  v = sendto(scheduler_fd_sec, (void*)&tx_p, sizeof(tf_runtime_packet), 0,
             (struct sockaddr*)&runtime_addr, sizeof(runtime_addr));
  return v;
}

int TfScheduler::SendPacketToRuntimeSecSocket(tf_initialization_packet& tx_p,
                                     struct sockaddr_un& runtime_addr) {
  int v;
  v = sendto(scheduler_fd_sec, (void*)&tx_p, sizeof(tf_initialization_packet), 0,
             (struct sockaddr*)&runtime_addr, sizeof(runtime_addr));
  return v;
}

int TfScheduler::ReceivePacketFromRuntime(tf_packet& rx_p,
                                          struct sockaddr_un& runtime_addr) {
  int v;
  v = recvfrom(scheduler_fd, &rx_p, sizeof(tf_packet), 0,
               (struct sockaddr*)&runtime_addr, (socklen_t*)&addr_size);
  return v;
}

int TfScheduler::ReceivePacketFromRuntime(tf_runtime_packet& rx_p,
                                          struct sockaddr_un& runtime_addr) {
  int v;
  v = recvfrom(scheduler_fd, &rx_p, sizeof(tf_runtime_packet), 0,
               (struct sockaddr*)&runtime_addr, (socklen_t*)&addr_size);
  return v;
}

int TfScheduler::ReceivePacketFromRuntimeMultiplex(tf_runtime_packet& rx_p,
                                            struct sockaddr_un& runtime_addr,
                                            int max_fd, fd_set& read_fds){
  std::cout << "multiplex wait" << "\n";
  int activity = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
  if (activity < 0) {
    std::cout << "select error" << "\n";
    return -1;
  }

  // read from main inference thread
  if (FD_ISSET(scheduler_fd, &read_fds)) {
    std::cout << "read from main inference thread" << "\n";
    if (ReceivePacketFromRuntime(rx_p, runtime_addr) == -1) {
      std::cout << "Receive failed"
                << "\n";
      return -1;
    }
    return 1;
  }

  // read from secondary inference thread
  if (FD_ISSET(scheduler_fd_sec, &read_fds)) {
    std::cout << "read from secondary inference thread" << "\n";
    return 1;
    if (ReceivePacketFromRuntimeSecSocket(rx_p, runtime_addr) == -1) {
      std::cout << "Receive failed"
                << "\n";
      return -1;
    }
  }
  std::cout << "select done but read error" << "\n";
  return -1;
}

// option1 한번이라도 다음주에 하게 해달라고 비빈다. 
// option2 2025 후기를 보겠다고 하고 더 논의한다. -> 회사 생활을 할지, 랩 인턴 생활을 할지 본다. 
// option3 그냥 알겠다고 하고 접는다 -> 일단 회사 가고 후기 입학을 노린다. 
// option4 

int TfScheduler::ReceivePacketFromRuntime(tf_initialization_packet& rx_p,
                                          struct sockaddr_un& runtime_addr) {
  int v;
  v = recvfrom(scheduler_fd, &rx_p, sizeof(tf_initialization_packet), 0,
               (struct sockaddr*)&runtime_addr, (socklen_t*)&addr_size);
  return v;
}

// Experimental feature to use secondary socket.
int TfScheduler::ReceivePacketFromRuntimeSecSocket(tf_runtime_packet& rx_p,
                                          struct sockaddr_un& runtime_addr) {
  int v;
  v = recvfrom(scheduler_fd_sec, &rx_p, sizeof(tf_runtime_packet), 0,
               (struct sockaddr*)&runtime_addr, (socklen_t*)&addr_size);
  if(!rx_p.is_secondary_socket) {v = -1;}
  return v;
}

// Experimental feature to use secondary socket.
int TfScheduler::ReceivePacketFromRuntimeSecSocket(tf_initialization_packet& rx_p,
                                          struct sockaddr_un& runtime_addr) {
  int v;
  v = recvfrom(scheduler_fd_sec, &rx_p, sizeof(tf_initialization_packet), 0,
               (struct sockaddr*)&runtime_addr, (socklen_t*)&addr_size);
  if(!rx_p.is_secondary_socket) {v = -1;}
  return v;
}

void TfScheduler::Work() {
  monitor = new LiteSysMonitor();
  bool run = true;

  // Experimantal flag for the seperation of init packet and runtime packet.
  // Using init packet on runtime may cost huge communication overhead(packet payload is big).
  // Scheduler runs only single TFLite runtime.
  bool init = true;
  
  // Experimental feature that enables semi-asynchronous control of computing resources(CPU, GPU).
  // Initalizes secondary uds socket.
  // THIS IS EXPERIMENTAL
  {
    //receive 
    tf_initialization_packet rx_init_packet;
    struct sockaddr_un runtime_addr;
    memset(&rx_init_packet, 0, sizeof(tf_initialization_packet));
    if (ReceivePacketFromRuntimeSecSocket(rx_init_packet, runtime_addr) == -1) {
      std::cout << "Secondary socket receive failed"
                << "\n";
      return;
    }
    tf_initialization_packet tx_init_packet;
    //send
    if (SendPacketToRuntimeSecSocket(tx_init_packet, runtime_addr) == -1) {
      std::cout << "Secondary socket : " << runtime_addr.sun_path << " "
                << runtime_addr.sun_family << "\n";
      printf("errno : %d \n", errno);
      return;
    }
    std::cout << "Secondary socket connected" << "\n";
    //ok
  }
  // Set fd for multiplexing
  fd_set read_fds;
  int max_fd = std::max(scheduler_fd, scheduler_fd_sec);
  // struct timeval timeout;
  // timeout.tv_sec = 10;
  // timeout.tv_usec = 0;

  while (run) {
    // tf_initialization_packet rx_packet;
    tf_initialization_packet rx_init_packet;
    tf_runtime_packet rx_runtime_packet;
    struct sockaddr_un runtime_addr;
    RuntimeState state = RuntimeState::INITIALIZE;
    int id = 0;
    if(init){
      memset(&rx_init_packet, 0, sizeof(tf_initialization_packet));
      if (ReceivePacketFromRuntime(rx_init_packet, runtime_addr) == -1) {
        std::cout << "Receive failed"
                  << "\n";
        return;
      }
      state = static_cast<RuntimeState>(rx_init_packet.runtime_current_state);
      id = static_cast<int>(rx_init_packet.runtime_id);
    }else{
      FD_ZERO(&read_fds);
      FD_SET(scheduler_fd, &read_fds);
      FD_SET(scheduler_fd_sec, &read_fds);
      
      // TODO change to socket multiplexing
      // if (ReceivePacketFromRuntime(rx_runtime_packet, runtime_addr) == -1) {
      //   std::cout << "Receive failed"
      //             << "\n";
      //   return;
      // }
      memset(&rx_runtime_packet, 0, sizeof(tf_runtime_packet));
      if (ReceivePacketFromRuntimeMultiplex(rx_runtime_packet, runtime_addr,
                                            max_fd, read_fds) == -1) {
        std::cout << "Receive multiplex failed"
                  << "\n";
        return;
      }
    
    
      state = static_cast<RuntimeState>(rx_runtime_packet.runtime_current_state);
      id = static_cast<int>(rx_runtime_packet.runtime_id);
    }



    // [VLS Todo] need packet check??
    // tf_packet rx_packet;
    #ifdef debug
      std::cout << "Received packet from runtime " << rx_packet.runtime_id <<
      "\n";
    #endif
    // do next work by received runtime state.
    switch (state) {
      case RuntimeState::INITIALIZE: {
        bool is_already_registered = false;
        std::cout << "Runtime init"
                  << "\n";
        for (auto runtime : runtimes) {
          if (runtime->id == id) {
            std::cout << "Runtime " << runtime->id << " already registered."
                      << "\n";
            is_already_registered = true;
            break;
          }
        }if(!is_already_registered){
          // initializing new_runtime
          runtime_* new_runtime = new runtime_;
          new_runtime->id = runtimes_created;
          runtimes_created++;

          new_runtime->addr.sun_family = runtime_addr.sun_family;
          strcpy(new_runtime->addr.sun_path, runtime_addr.sun_path);
          tf_initialization_packet tx_packet;
          //tf_packet tx_packet;
          memset(&tx_packet, 0, sizeof(tf_initialization_packet));
          tx_packet.runtime_id = new_runtime->id;
          tx_packet.runtime_next_state = RuntimeState::NEED_PROFILE;

          if (SendPacketToRuntime(tx_packet, runtime_addr) == -1) {
            std::cout << "Sending packet to " << new_runtime->id << " Failed"
                      << "\n";
            std::cout << "sock : " << runtime_addr.sun_path << " "
                      << runtime_addr.sun_family << "\n";
            printf("errno : %d \n", errno);
            return;
          }
          runtimes.push_back(new_runtime);
          std::cout << "Registered new runtime " << new_runtime->id << " \n";
        }
        break;
      }
      case RuntimeState::NEED_PROFILE: {
        tf_initialization_packet tx_packet;
        
        RefreshRuntimeState(rx_init_packet);
        
        // store multi-level params in vector and send it to runtime.
        CreatePartitioningPlan(rx_init_packet, subgraph_params_sched, subgraph_params_runtime);
        // Creates multi-level subgraph structure in scheduler.
        CreateGraphofSubgraphs(id, subgraph_params_sched);
        tx_packet.runtime_id = id;
        tx_packet.level = 0; // sends parameter starting with level 0.
        if(tx_packet.level != rx_init_packet.level)
          tx_packet.level = rx_init_packet.level;
        // check if parameter transmit done.
        if(subgraph_params_runtime.size() == tx_packet.level){
          std::cout << "parameter transmission complete" << "\n";
          tx_packet.runtime_next_state = RuntimeState::SUBGRAPH_CREATE;
          tx_packet.level = 0; // parameter transmission finished.
        }else{
          std::cout << "send param level " << tx_packet.level << "\n";
          tx_packet.runtime_next_state = RuntimeState::NEED_PROFILE;
          // load packet payload with subgraph partitioning param.
          // size check (param < 10000)
          if(subgraph_params_runtime[tx_packet.level].size() > TF_P_PLAN_LENGTH){
            std::cout << "Parameter size exceed 10000, cannot handle." << "\n";
            exit(-1);
          }else{
          // copy
          std::copy(subgraph_params_runtime[tx_packet.level].begin(), 
                    subgraph_params_runtime[tx_packet.level].end(),
                    tx_packet.partitioning_plan);
          // send
          }
        }
        // [VLS Todo] consider this part iterate multiple time?
        if (SendPacketToRuntime(tx_packet, runtime_addr) == -1) {
          std::cout << "sock : " << runtime_addr.sun_path << " "
                    << runtime_addr.sun_family << "\n";
          printf("errno : %d \n", errno);
          return;
        }
        std::cout << "Need profile done" << "\n";
        break;
      }
      case RuntimeState::SUBGRAPH_CREATE: {
        RefreshRuntimeState(rx_init_packet);
        
        tf_initialization_packet tx_packet;
        // tf_packet tx_packet;
        tx_packet.runtime_id = id;

        // not done
        std::cout << "Prepare runtime " << "\n";
        for(int i=0; i<10; ++i){
          std::cout << rx_init_packet.subgraph_ids[0][i] << " ";
        }
        std::cout << "\n";
        // [VLS Todo] fix to call this function multiple time.
        PrepareRuntime(rx_init_packet);
        std::cout << "Prepare runtime done" << "\n";
        if(rx_init_packet.level == subgraph_params_runtime.size()-1){
          std::cout << "Send invoke state" << "\n";
          tx_packet.runtime_next_state = RuntimeState::INVOKE_;
          PrintGraph(id);
                             init = false;
        }else{
          tx_packet.runtime_next_state = RuntimeState::SUBGRAPH_CREATE;
        }

        if (SendPacketToRuntime(tx_packet, runtime_addr) == -1) {
          std::cout << "sock : " << runtime_addr.sun_path << " "
                    << runtime_addr.sun_family << "\n";
          printf("errno : %d \n", errno);
          return;
        }
        //Set initialization flag false.
        break;
      }
      case RuntimeState::INVOKE_: {
        // [VLS Todo] use runtime packet here.
        RefreshRuntimeState(rx_runtime_packet);
        tf_runtime_packet tx_packet;
        // tf_packet tx_packet;
        tx_packet.runtime_id = id;
        tx_packet.runtime_next_state = RuntimeState::INVOKE_;

        std::pair<int, int> next_subgraph_to_invoke;
        next_subgraph_to_invoke = SearchNextSubgraphtoInvoke(rx_runtime_packet);
        tx_packet.subgraph_ids[0][0] = next_subgraph_to_invoke.first;
        tx_packet.subgraph_ids[1][0] = next_subgraph_to_invoke.second;

        if (SendPacketToRuntime(tx_packet, runtime_addr) == -1) {
          std::cout << "sock : " << runtime_addr.sun_path << " "
                    << runtime_addr.sun_family << "\n";
          printf("errno : %d \n", errno);
          return;
        }
        break;
      }
      case RuntimeState::TERMINATE: {
        std::cout << "Scheduler got terminate signal"
                  << "\n";
        run = false;
        break;
      }
      default:
        break;
    }
  }
  return;
}

// [VLS todo]
// change to read multiple subgraph params safely.
void TfScheduler::OpenPartitioningParams(std::vector<std::string>& param_file_names) {
  for(auto file_name : param_file_names){
    std::cout << "Read param " << file_name << "\n";
    yaml_param_files.push_back(file_name);
    // std::fstream* new_file = new std::fstream;
    // new_file->open(file_name, std::fstream::in);
    // param_files.push_back(new_file);
  }
}

// [VLS] Todo - change to search subgraph in multi-level.
std::pair<int, int> TfScheduler::SearchNextSubgraphtoInvoke(
    tf_runtime_packet& rx_packet) {
  std::pair<int, int> next_subgraphs_to_invoke;
  int runtime_id = rx_packet.runtime_id;
  runtime_* runtime = nullptr;
  for (int i = 0; i < runtimes.size(); ++i) {
    if (runtimes[i]->id == runtime_id) runtime = runtimes[i];
  }
  if (runtime == nullptr) {
    std::cout << "Cannot find matching runtime in SearchNextSubgraphtoInvoke()"
              << "\n";
    exit(-1);
  }
  ////////////////////////////////////////////////////////////////////////
  //  New logic for multi-level subgraphs.
  //  0. monitor -> level change flag (increase, decrease)
  //  1.level selection
  //     if(level change)
  //        if(recovery possible) -> recovery
  //        not -> level change, flag initialize
  //  2.subgraph selection
  //     - ordinary selection
  //     - recovery selection
  ////////////////////////////////////////////////////////////////////////

  int gpu_thresh = 70;
  int cpu_thresh = 70;
  int level = 0; // slow start
  bool first_and_last_graph = false;
  subgraph_node* root_graph = runtime->graphs[level]->root;
  subgraph_node* prev_invoked_subgraph = nullptr;
  subgraph_node* prev_base_subgraph = nullptr;
  // std::cout << "cur_graph : " << rx_packet.cur_subgraph << "\n";
  // if(rx_packet.cur_subgraph != -1){
  //   printf("main latency: %.6f, sub latency: %.6f\n", rx_packet.main_interpret_response_time,
  //           rx_packet.sub_interpret_response_time);
  // }
  if (rx_packet.cur_subgraph == -1) {  // first invoke
    // std::cout << "first invoke" << "\n";
    // search graph struct for optimal invokable subgraph.
    // and return it.
    // ISSUE(dff3f) : Right after GPU kernel initialization, gpu utilization
    // ratio raises shortly.

    // Set root graph for first graph.
    prev_invoked_subgraph = root_graph;
    prev_base_subgraph = root_graph;
  } else {
    // Search and return prev invoked subgraph with it's id.
    prev_invoked_subgraph =
        SearchAndReturnNodeWithID(root_graph, rx_packet.cur_subgraph);
    prev_base_subgraph = prev_invoked_subgraph;
    while (prev_base_subgraph->up != nullptr) {
      prev_base_subgraph = prev_base_subgraph->up;
    }
  }

  // In case only one subgraph exists.
  if (runtime->graphs[level]->nodes.size() == 1) {
    next_subgraphs_to_invoke.first = runtime->graphs[level]->nodes[0]->subgraph_id;
    next_subgraphs_to_invoke.second = runtime->graphs[level]->nodes[0]->co_subgraph_id;

    // case of final subgraph..
    if (rx_packet.cur_subgraph == runtime->graphs[level]->nodes[0]->subgraph_id) {
      next_subgraphs_to_invoke.first = -1;
      next_subgraphs_to_invoke.second = -1;
    }
    std::cout << "one subgraph" << "\n";
    return next_subgraphs_to_invoke;
  }

  subgraph_node* next_base_subgraph = nullptr;
  subgraph_node* next_subgraph_to_invoke = nullptr;

  // case of final subgraph
  #ifdef single_level_motivation
    if(rx_packet.cur_subgraph != -1){
      std::cout << "motiv final subgraph" << "\n";
      next_subgraphs_to_invoke.first = -1;
      next_subgraphs_to_invoke.second = -1;  
      return next_subgraphs_to_invoke;
    }else{
      next_base_subgraph = root_graph;
    }
  #endif
  #ifndef single_level_motivation
    if (prev_base_subgraph->right == nullptr) {
      next_subgraphs_to_invoke.first = -1;
      next_subgraphs_to_invoke.second = -1;
      std::cout << "final graph" << "\n";
      return next_subgraphs_to_invoke;
    }  // case of first subgraph
    else if (rx_packet.cur_subgraph == -1) {
      next_base_subgraph = root_graph;
    } else {
      next_base_subgraph = prev_base_subgraph->right;
    }
  #endif`

  int next_resource_plan = -1;
  next_subgraph_to_invoke = next_base_subgraph;
  // std::cout << "level " << level  << " subgraph " << 
  //           next_subgraph_to_invoke->subgraph_id << "\n";
  // ISSUE ,MUST FIX (07b4f) : Consider the gpu utilization ratio delay.
  // NEED_REFACTOR (02634) : Must change to use obvious resource type.
  int next_cpu_resource = 0;
  float gpu_util = monitor->GetGPUUtil();
  float cpu_util = monitor->GetCPUUtil();

  /* [IMPT] Utilization based resource allocation part */
  // std::cout << "CPU : " << cpu_util << " GPU : " << gpu_util << "\n"; 
  printf("rx %.6f profiled %.6f \n",rx_packet.main_interpret_response_time ,prev_invoked_subgraph->average_latency);
  if(rx_packet.main_interpret_response_time > prev_invoked_subgraph->average_latency * 1.2){
    // printf("rx %d profiled %d \n",rx_packet.cur_subgraph ,prev_invoked_subgraph->subgraph_id);
    // 느려진 비율 만큼 느려질 것이라고 예상하면 되지않나???
    std::cout << "resource" << prev_invoked_subgraph->resource_type << "contention!" << "\n";
    std::cout << "GPU " << gpu_util << " CPU " << cpu_util << "\n";
    if(prev_invoked_subgraph->resource_type == 3 && gpu_util < 50){ //cpu
      std::cout << "use gpu \n";
      gpu_usage_flag = true;
      cpu_usage_flag = false;
    }else if(prev_invoked_subgraph->resource_type == 1 && cpu_util < 50){
      std::cout << "use cpu \n";
      gpu_usage_flag = false;
      cpu_usage_flag = true;
    }
  }
  if(gpu_usage_flag){
    next_resource_plan = 1;
  }
  if(cpu_usage_flag){
    next_resource_plan = 3;
  }
  // if (gpu_util == 0 && cpu_util == 400) {
  //   // Use CPU
  //   next_resource_plan = TF_P_PLAN_GPU;
  //   std::cout << "USE GPU" << "\n";
  // } else if (gpu_util == 100 && cpu_util == 0) {
  //   // Use GPU
  //   next_resource_plan = TF_P_PLAN_CPU_XNN;
  //   std::cout << "USE CPU" << "\n";
  // } else if (gpu_util == 100 && cpu_util == 200) {
  //   // Use Co-execution
  //   cpu_usage_flag = true;
  //   std::cout << "USE CPU200" << "\n";
  // } else {
  //   // base plan
  //   cpu_usage_flag = false;
  //   std::cout << "USE BASE" << "\n";
  //   next_resource_plan = next_base_subgraph->resource_type;
  // }
  
  // base code.
  //next_resource_plan = next_base_subgraph->resource_type;
  
  /* [IMPT] */

  // TODO (f85fa) : Fix graph searching, especially in co-execution.
  // Search for matching subgraph.
  while (next_subgraph_to_invoke != nullptr) {
    if (next_subgraph_to_invoke->resource_type == next_resource_plan) {
      break;
    }
    if (next_subgraph_to_invoke->down != nullptr) {
      next_subgraph_to_invoke = next_subgraph_to_invoke->down;
    } else {
      next_subgraph_to_invoke = next_base_subgraph;
      break;
    }
  }

  // std::cout << "set next_subgraph_to_invoke id " <<
  // next_subgraph_to_invoke->subgraph_id << "\n"; std::cout << "set
  // next_subgraph_to_invoke co id " << next_subgraph_to_invoke->co_subgraph_id
  // << "\n"; std::cout << "set next_subgraph_to_invoke resource_type " <<
  // next_subgraph_to_invoke->resource_type << "\n";
  next_subgraphs_to_invoke.second = next_subgraph_to_invoke->co_subgraph_id;
  next_subgraphs_to_invoke.first = next_subgraph_to_invoke->subgraph_id;
  next_subgraphs_to_invoke.second = next_subgraph_to_invoke->co_subgraph_id;
  return next_subgraphs_to_invoke;
}

// [VLS Todo] call this function multi-level safely.
void TfScheduler::PrepareRuntime(tf_initialization_packet& rx_packet) {
  int runtime_id = rx_packet.runtime_id;
  int level = rx_packet.level;
  std::cout << "Register level " << level << " subgraphs in scheduler" << "\n";
  runtime_* runtime = nullptr;
  for (int i = 0; i < runtimes.size(); ++i) {
    if (runtimes[i]->id == runtime_id) runtime = runtimes[i];
  }
  std::cout << "Runtime " << runtime_id << " has " << runtime->graphs.size() << " levels of subgraphs" << "\n";
  for(int i=0; i<runtime->graphs.size(); ++i){
    // std::cout << "level " << runtime->graphs[i]->level << "\n"; 
    std::cout << "level " << i << " has " << runtime->graphs[i]->nodes.size() << " subgraphs " << "\n"; 
  }
  // TODO(28caeaf) : Read the subgraph ids from packet and make it as
  // linked-list?
  if (runtime == nullptr) {
    std::cout << "Cannot find matching runtime in PrepareRuntime()"
              << "\n";
    exit(-1);
  }
  if(runtime->graphs.size() < level){
    std::cout << "Runtime " << runtime_id << " doesn't have " <<
                 "subgraph level " << level << "\n";
    exit(-1);
  }
  std::queue<int> co_subgraph_ids;
  std::vector<int> recv_subgraph_ids;
  int num_co_subs = 0;
  std::cout << "Asdf1" << "\n";
  
  for(int i=0; i<10; ++i){
    std::cout << rx_packet.subgraph_ids[0][i] << " ";
  }
  std::cout << "\n";

  int idx = 0;
  while (rx_packet.subgraph_ids[0][idx] != -1) {
    // std::cout << rx_packet.subgraph_ids[0][idx] << "\n";
    recv_subgraph_ids.push_back(rx_packet.subgraph_ids[0][idx]);
    idx++;
  }

  idx = 0;
  std::cout << "got main ids" << "\n";
  while (rx_packet.subgraph_ids[1][idx] != -1) {
    // std::cout << rx_packet.subgraph_ids[1][idx] << "\n";
    num_co_subs++;
    co_subgraph_ids.push(rx_packet.subgraph_ids[1][idx]);
    idx++;
  }
  std::cout << "got sub ids" << "\n";

  // Register main subgraphs
  for (int i = 0; i < recv_subgraph_ids.size(); ++i) {
    runtime->graphs[level]->nodes[i]->subgraph_id = recv_subgraph_ids[i];
  }
  // Register Co subgraphs
  // MUST FIX(b6582) : co-subgraph does not always exist at the end of whole
  // graph structure.
  idx = 0;
  while (!co_subgraph_ids.empty()) {
    if (runtime->graphs[level]->nodes[idx]->resource_type == 2 ||
        runtime->graphs[level]->nodes[idx]->resource_type == 4) {
      runtime->graphs[level]->nodes[idx]->co_subgraph_id = co_subgraph_ids.front();
      co_subgraph_ids.pop();
    }
    idx++;
  }

  if ((recv_subgraph_ids.size() - num_co_subs) != runtime->graphs[level]->nodes.size()) {
    std::cout << "Subgraph ids from runtime and existing graph"
              << " does not match"
              << "\n";
    return;
  }
}

void TfScheduler::CreateGraphofSubgraphs(int id, 
                                        std::vector<std::vector<int>>& subgraph_params_sched) {
  for (int runtime_idx = 0; runtime_idx < runtimes.size(); ++runtime_idx) {
    if (runtimes[runtime_idx]->id == id) {
      runtime_* working_runtime = runtimes[runtime_idx];
      if (!working_runtime->graphs.empty()) {
        std::cout << "Scheudler: Runtime " << working_runtime->id
                  << " already has graph."
                  << "\n";
        return;
      }
      for(int level = 0; level<subgraph_params_sched.size(); ++level){ // iterate subgraph-level
        // Create new graph structure for current runtime.
        subgraph_graph* new_level_subgraph = new subgraph_graph;
        new_level_subgraph->runtime_id = id;
        new_level_subgraph->level = level;
        
        int working_idx = 0;
        int current_value = subgraph_params_sched[level][working_idx]; // Get first value of partitioning plan
        // std::cout << "cur val " << current_value << "\n";
        subgraph_node* new_node;
        int start_node, end_node, partitioning_ratio, resource_type, cpu_util, gpu_util;
        float latency;
        bool start_node_flag = true;
        bool root_graph = true;
        while(current_value != PART_PARM_SEP_ENDP){  
          if(current_value == PART_PARM_SEP_OP){ // means end of node subset 
                                                 // (initialize new subgraph_node)
            // end_node = tx_packet.partitioning_plan[working_idx - 1];
            end_node = subgraph_params_sched[level][working_idx-1];
            start_node_flag = true; // refresh used flag
            working_idx += 5;
          }else if(current_value == PART_PARM_SEP_RESROURCE){ // means end of resource & partitioning plan 
                                                        // (add current subgraph node)
            resource_type = subgraph_params_sched[level][working_idx - 5];
            partitioning_ratio = subgraph_params_sched[level][working_idx - 4];
            std::cout << "register " << subgraph_params_sched[level][working_idx - 3] << "\n";
            std::cout << "start op " << start_node << " end op " << end_node << "\n";
            latency = float(subgraph_params_sched[level][working_idx - 3]) / 1000000.0;
            cpu_util = subgraph_params_sched[level][working_idx - 2];
            gpu_util = subgraph_params_sched[level][working_idx - 1];
            if(root_graph){ // add this graph to root graph.
              root_graph = false;
              new_node->rank = 0;
              new_node->partitioning_ratio = partitioning_ratio;
              new_node->resource_type = resource_type;
              new_node->node_start = start_node;
              new_node->node_end = end_node;
              new_node->average_latency = latency;
              new_node->cpu_utilization_require = cpu_util;
              new_node->gpu_utilization_require = gpu_util;
              new_level_subgraph->root = new_node;
              new_level_subgraph->nodes.push_back(new_node);
              std::cout << "Create root node" << "\n";
            }else{ // add this graph to leaf graph.
              if(!AddSubgraphtoGraph(new_level_subgraph,
                                    start_node,
                                    end_node,
                                    resource_type,
                                    partitioning_ratio,
                                    latency,
                                    cpu_util,
                                    gpu_util)){
                std::cout << "AddSubgraphtoGraph ERROR" << "\n";
                return;
              }
            }
          }else if(current_value != PART_PARM_SEP_SUBG){ // means node subset (add)
            if(start_node_flag){
              new_node = new subgraph_node;
              start_node = current_value;
              // std::cout << "start node " << current_value << "\n";
              start_node_flag = false;
              if(working_idx == 0) { root_graph = true; }
            }
          }
          working_idx++;
          // current_value = tx_packet.partitioning_plan[working_idx];
          current_value = subgraph_params_sched[level][working_idx];
          // std::cout << "cur val " << current_value << "\n";
        }// subgraph interation
        std::cout << "push new level" << "\n";
        working_runtime->graphs.push_back(new_level_subgraph);
      }// subgraph level iteration
    } // runtime iteration
  }
  return;
}

bool TfScheduler::AddSubgraphtoGraph(subgraph_graph* graph, int s_node,
                                     int e_node, int resource_type,
                                     int partitioning_ratio, float latency,
                                     int cpu_util, int gpu_util) {
  subgraph_node* pointer = graph->root;
  subgraph_node* new_node;
  int new_rank = 0;
  pointer = SearchAndReturnBaseNode(pointer, s_node, e_node, new_rank);

  new_node = new subgraph_node;
  new_node->node_start = s_node;
  new_node->node_end = e_node;
  new_node->resource_type = resource_type;
  new_node->partitioning_ratio = partitioning_ratio;
  new_node->average_latency = latency;
  new_node->cpu_utilization_require = cpu_util;
  new_node->gpu_utilization_require = gpu_util;
  if (new_rank ==
      0) {  // if rank is 0, just add node at the end of graph. (left to right)
    pointer->right = new_node;
    new_node->left = pointer;
    new_node->rank = new_rank;
  } else {
    pointer->down = new_node;
    new_node->up = pointer;
    new_node->rank = new_rank;
  }
  graph->nodes.push_back(new_node);
  std::cout << "create new node" << "\n";
  return true;
}

// Search the graph structure down to up & left to right.
subgraph_node* TfScheduler::SearchAndReturnBaseNode(subgraph_node* node,
                                                    int s_node, int e_node,
                                                    int& new_rank) {
  std::queue<subgraph_node*> node_q;
  node_q.push(node);
  while (!node_q.empty()) {
    if (node->down == nullptr && node->node_start == s_node &&
        node->node_end == e_node) {
      new_rank++;
      return node;
    }
    while (node->down != nullptr) {
      node = node->down;
      new_rank++;
      if (node->down == nullptr && node->node_start == s_node &&
          node->node_end == e_node) {
        new_rank++;
        return node;
      }
    }
    node = node_q.front();
    node_q.pop();
    if (node->right != nullptr) {
      node = node->right;
      new_rank = node->rank;
      node_q.push(node);
    } else {
      return node;
    }
  }
}

subgraph_node* TfScheduler::SearchAndReturnNodeWithID(subgraph_node* root,
                                                      int id) {
  std::queue<subgraph_node*> node_q;
  node_q.push(root);
  subgraph_node* node = nullptr;
  while (!node_q.empty()) {
    node = node_q.front();
    node_q.pop();
    if (node->right != nullptr) node_q.push(node->right);
    while (node != nullptr) {
      if (node->subgraph_id == id) return node;
      node = node->down;
    }
  }
  std::cout << "Cannot find matching subgraph in graph"
            << "\n";
  return nullptr;
}

void TfScheduler::PrintGraph(int runtime_id) {
  runtime_* runtime = nullptr;
  for (int i = 0; i < runtimes.size(); ++i) {
    if (runtimes[i]->id == runtime_id) runtime = runtimes[i];
  }

  std::cout << "Prints subgraphs in runtime " << runtime->id
            << "\n";
  for(int level = 0; level < runtime->graphs.size(); ++level){
    std::cout << "Subgraph level : " << level << "\n";
    for (int i = 0; i < runtime->graphs[level]->nodes.size(); ++i) {
      std::cout << "Main subgraph ID " << runtime->graphs[level]->nodes[i]->subgraph_id
                << " ";
      std::cout << "Co subgraph ID " << runtime->graphs[level]->nodes[i]->co_subgraph_id
                << " ";
      std::cout << "RANK " << runtime->graphs[level]->nodes[i]->rank << "\n";
    }
  }  
}

bool TfScheduler::CheckAllRuntimesReady() {
  if (runtimes.size() != 2) {
    return false;
  }
  for (int i = 0; i < runtimes.size(); ++i) {
    if (runtimes[i]->state != RuntimeState::INVOKE_) return false;
  }
  return true;
}

void TfScheduler::RefreshRuntimeState(tf_packet& rx_p) {
  for (int i = 0; i < runtimes.size(); ++i) {
    if (rx_p.runtime_id == runtimes[i]->id) {
      runtimes[i]->state =
          static_cast<RuntimeState>(rx_p.runtime_current_state);
    }
  }
}

void TfScheduler::RefreshRuntimeState(tf_runtime_packet& rx_p) {
  for (int i = 0; i < runtimes.size(); ++i) {
    if (rx_p.runtime_id == runtimes[i]->id) {
      runtimes[i]->state =
          static_cast<RuntimeState>(rx_p.runtime_current_state);
    }
  }
}

void TfScheduler::RefreshRuntimeState(tf_initialization_packet& rx_p) {
  for (int i = 0; i < runtimes.size(); ++i) {
    if (rx_p.runtime_id == runtimes[i]->id) {
      runtimes[i]->state =
          static_cast<RuntimeState>(rx_p.runtime_current_state);
    }
  }
}

void TfScheduler::ReleaseResource(ResourceType type) {
  switch (type) {
    case ResourceType::CPU:
      cpu_usage_flag = false;
      break;

    case ResourceType::GPU:
      gpu_usage_flag = false;
      break;

      // case ResourceType::CPUGPU :
      //   cpgpu_usage_flag = false;
      //   break;

    default:
      break;
  }
  return;
}

void TfScheduler::PrintRuntimeStates() {
  std::cout << "===================================";
  std::cout << "TfScheduler has " << runtimes.size() << " runtimes"
            << "\n";
  for (int i = 0; i < runtimes.size(); ++i) {
    std::cout << "===================================";
    std::cout << "Runtime ID : " << runtimes[i]->id << "\n";
    std::cout << "Runtime State : " << runtimes[i]->state << "\n";
    std::cout << "Socket path :" << runtimes[i]->addr.sun_path << "\n";
  }
}

// [VLS Todo] change to create multi-level subgraph partitioning plan.
// deprecated.
// deprecated.
// deprecated.
void TfScheduler::CreatePartitioningPlan(tf_initialization_packet& rx_p,
                          std::vector<std::vector<int>>& subgraph_params) {
  if(!subgraph_params.empty()){
    return; // parameter already created.
  }
  int ops = 0; // Count number of operators in given model.
  for (int i = 0; i < 1000; ++i) {
    if (rx_p.latency[i] == -1)
      ops++;
    else
      break;
  }
  std::cout << "Runtime [" << rx_p.runtime_id << "] has " << ops
            << " ops in model"
            << "\n";
      
  int level = 0; // subgaph level
  for(auto param_file : param_files){ // iterate each parameter files.
    subgraph_params.push_back(std::vector<int>()); // create empty vector for param.
    std::string line, token;
    int arg = 0;
    int line_iter = 0; // line reader
    int plan_idx = 0;
    bool seperator_flag = false;
    // get line
    if(!param_file->is_open()){
      std::cout << "Scheduler ERROR : Param file is not opened" << "\n";
      exit(-1);
    }
    // get param line by line
    while (std::getline(*param_file, line)) {
      switch (line_iter)
      {
      case 0:{
        if(line == "*"){ // subgraph candidate set end flag
          subgraph_params[level].push_back(PART_PARM_SEP_SUBG);
        }else if(line == "-"){ // subgraph level end flag
          subgraph_params[level].push_back(PART_PARM_SEP_ENDP);
        }else{ // operators
          std::stringstream s_stream(line); // parse line to stringstream.
          while(getline(s_stream, token, ' ')){
            arg = std::stoi(token);  
            subgraph_params[level].push_back(arg);
            line_iter = 1;
          }  
        }
        break;
      }
      case 1: // ratio
        subgraph_params[level].push_back(PART_PARM_SEP_OP);
        subgraph_params[level].push_back(std::stoi(line));
        line_iter = 2;
        break;
      case 2: // resource
        subgraph_params[level].push_back(std::stoi(line));
        subgraph_params[level].push_back(PART_PARM_SEP_RESROURCE);
        line_iter = 0;
        break;
      }
    } // param_file parsing iter.
    level++;
  } // param_file iter.
  // for debugging purpose.
  int lev = 0;
  for(auto sub_param : subgraph_params){
    std::cout << "level : " << lev << "\n";
    for(auto i : sub_param){
      std::cout << i << " ";
    }
    std::cout << "\n";
    lev++;
  }
  return;
}

void TfScheduler::CreatePartitioningPlan(tf_initialization_packet& rx_p,
                                  std::vector<std::vector<int>>& subgraph_params_sched,
                                  std::vector<std::vector<int>>& subgraph_params_runtime){
  if(!subgraph_params_runtime.empty()){
    return; // parameter already created.
  }
  int ops = 0; // Count number of operators in given model.
  for (int i = 0; i < 1000; ++i) {
    if (rx_p.latency[i] == -1)
      ops++;
    else
      break;
  }
  std::cout << "Runtime [" << rx_p.runtime_id << "] has " << ops
            << " ops in model"
            << "\n";
  int level = 0; // subgraph level
  // A single yaml file contains parameters of single 'level' of subgraph. 
  for(auto param_file : yaml_param_files){
    subgraph_params_sched.push_back(std::vector<int>());
    subgraph_params_runtime.push_back(std::vector<int>());
    YAML::Node params = YAML::LoadFile(param_file);
    for(const auto& candidate_level_node : params){
      std::cout << "candidate" << candidate_level_node["candidate_level"].as<int>() << "\n";
      for(const auto& graph : candidate_level_node["graphs"]){
        std::cout << "graphs" << "\n";
        if(graph["op"]) {
          for(const auto& element : graph["op"]) { // parse operators
            subgraph_params_runtime[level].push_back(element.as<int>());
            subgraph_params_sched[level].push_back(element.as<int>());
          }
        }
        if(graph["resource"]) { // parse ratio
          subgraph_params_runtime[level].push_back(PART_PARM_SEP_OP); 
          subgraph_params_sched[level].push_back(PART_PARM_SEP_OP); 
          subgraph_params_runtime[level].push_back(graph["resource"].as<int>()); 
          subgraph_params_sched[level].push_back(graph["resource"].as<int>()); 
        }
        if(graph["ratio"]) { // parse resource
          subgraph_params_runtime[level].push_back(graph["ratio"].as<int>()); 
          subgraph_params_runtime[level].push_back(PART_PARM_SEP_RESROURCE); 

          subgraph_params_sched[level].push_back(graph["ratio"].as<int>()); 
          // runtime doesn't need latency and CPU,GPU util.
          if(graph["latency"]) { // parse latency
            // profiled latency has microsec precision (x.xxxxxx sec) 
            // to contain in inteager we multiply 1.000.000
            subgraph_params_sched[level].push_back(
              static_cast<int>(graph["latency"].as<float>() * 1000000)); 
          }
          if(graph["CPUutil"]){ // parse CPU utilization demand
            subgraph_params_sched[level].push_back(graph["CPUutil"].as<int>());
          }
          if(graph["GPUutil"]){ // parse GPU utilizationb demand
            subgraph_params_sched[level].push_back(graph["GPUutil"].as<int>());
          }
          subgraph_params_sched[level].push_back(PART_PARM_SEP_RESROURCE); 
        }
      }
      subgraph_params_runtime[level].push_back(PART_PARM_SEP_SUBG); 
      subgraph_params_sched[level].push_back(PART_PARM_SEP_SUBG); 
    }
    subgraph_params_runtime[level].push_back(PART_PARM_SEP_ENDP); 
    subgraph_params_sched[level].push_back(PART_PARM_SEP_ENDP); 
    level++;
  }
  int le = 0;
  std::cout << "scheduler param" << "\n";
  for(auto param : subgraph_params_sched){
    std::cout << "level " << le << " params" << "\n";
    for(auto params : param){
      std::cout << params << " ";
    }
    le++;
    std::cout << "\n";
  }
  le = 0;
  std::cout << "runtime param" << "\n";
  for(auto param : subgraph_params_runtime){
    std::cout << "level " << le << " params" << "\n";
    for(auto params : param){
      std::cout << params << " ";
    }
    le++;
    std::cout << "\n";
  }
}


TfScheduler::~TfScheduler() {
  std::cout << "Scheduler Terminated"
            << "\n";
};

}  // namespace tflite
