#include "tensorflow/lite/tf_scheduler.h"

namespace tflite {

TfScheduler::TfScheduler(){};

TfScheduler::TfScheduler(const char* uds_file_name,
                         std::vector<std::string>& param_file_names) {
  // delete if sock file already exists.
  if (access(uds_file_name, F_OK) == 0) unlink(uds_file_name);

  scheduler_fd = socket(PF_FILE, SOCK_DGRAM, 0);
  if (scheduler_fd == -1) {
    std::cout << "Socker create ERROR"
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
  cpu_util = new float;
  gpu_util = new float;

  OpenPartitioningParams(param_file_names);
  std::cout << "Scheduler initializaing done"
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

int TfScheduler::ReceivePacketFromRuntime(tf_initialization_packet& rx_p,
                                          struct sockaddr_un& runtime_addr) {
  int v;
  v = recvfrom(scheduler_fd, &rx_p, sizeof(tf_initialization_packet), 0,
               (struct sockaddr*)&runtime_addr, (socklen_t*)&addr_size);
  return v;
}

void TfScheduler::Work() {
  monitor = new LiteSysMonitor();
  bool run = true;

  // Temporal flag for the seperation of init packet and runtime packet.
  // Using init packet on runtime may cost huge communication overhead(packet payload is big).
  // Scheduler runs only single TFLite runtime.
  bool init = true;
  
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
      memset(&rx_runtime_packet, 0, sizeof(tf_initialization_packet));
      if (ReceivePacketFromRuntime(rx_runtime_packet, runtime_addr) == -1) {
        std::cout << "Receive failed"
                  << "\n";
        return;
      }
      state = static_cast<RuntimeState>(rx_runtime_packet.runtime_current_state);
      id = static_cast<int>(rx_runtime_packet.runtime_id);
    }
    // [VLS Todo] need packet check??
    // tf_packet rx_packet;
    #ifdef debug
      std::cout << "Recieved packet from runtime " << rx_packet.runtime_id <<
      "\n";
    #endif
    // do next work by received runtime state.
    switch (state) {
      case RuntimeState::INITIALIZE: {
        std::cout << "Runtime init"
                  << "\n";
        for (auto runtime : runtimes) {
          if (runtime->id == id) {
            std::cout << "Runtime " << runtime->id << " already registered."
                      << "\n";
            break;
          }
        }
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
        break;
      }
      case RuntimeState::NEED_PROFILE: {
        tf_initialization_packet tx_packet;
        
        RefreshRuntimeState(rx_init_packet);
        
        // store multi-level params in vector and send it to runtime.
        std::vector<std::vector<int>> subgraph_params;
        CreatePartitioningPlan(rx_init_packet, subgraph_params);

        tx_packet.runtime_id = id;
        tx_packet.runtime_next_state = RuntimeState::SUBGRAPH_CREATE;

        CreateGraphofSubgraphs(id, subgraph_params);
        // CreateGraphofSubgraphs(tx_packet);

        // [VLS Todo] consider this part iterate multiple time?
        if (SendPacketToRuntime(tx_packet, runtime_addr) == -1) {
          std::cout << "sock : " << runtime_addr.sun_path << " "
                    << runtime_addr.sun_family << "\n";
          printf("errno : %d \n", errno);
          return;
        }
        break;
      }
      case RuntimeState::SUBGRAPH_CREATE: {
        RefreshRuntimeState(rx_init_packet);
        
        tf_initialization_packet tx_packet;
        // tf_packet tx_packet;
        tx_packet.runtime_id = id;
        tx_packet.runtime_next_state = RuntimeState::INVOKE_;

        // not done
        std::cout << "Prepare runtime " << "\n";
        // [VLS Todo] fix to get multi-level subgraphs.
        PrepareRuntime(rx_init_packet);
        std::cout << "Prepare runtime done" << "\n";
        PrintGraph(id);

        if (SendPacketToRuntime(tx_packet, runtime_addr) == -1) {
          std::cout << "sock : " << runtime_addr.sun_path << " "
                    << runtime_addr.sun_family << "\n";
          printf("errno : %d \n", errno);
          return;
        }
        //Set initialization flag false.
        init = false;
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
    std::fstream* new_file = new std::fstream;
    new_file->open(file_name, std::fstream::in);
    param_files.push_back(new_file);
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
  //
  //
  //
  //
  ////////////////////////////////////////////////////////////////////////

  // int gpu_thresh = 70;
  // int cpu_thresh = 70;
  // subgraph_node* root_graph = runtime->graph->root;
  // subgraph_node* prev_invoked_subgraph = nullptr;
  // subgraph_node* prev_base_subgraph = nullptr;
  // if (rx_packet.cur_subgraph == -1) {  // first invoke
  //   // search graph struct for optimal invokable subgraph.
  //   // and return it.
  //   // ISSUE(dff3f) : Right after GPU kernel initialization, gpu utilization
  //   // ratio raises shortly.

  //   // Set root graph for first graph.
  //   prev_invoked_subgraph = root_graph;
  //   prev_base_subgraph = root_graph;
  // } else {
  //   // Search and return prev invoked subgraph with it's id.
  //   prev_invoked_subgraph =
  //       SearchAndReturnNodeWithID(root_graph, rx_packet.cur_subgraph);
  //   prev_base_subgraph = prev_invoked_subgraph;
  //   while (prev_base_subgraph->up != nullptr) {
  //     prev_base_subgraph = prev_base_subgraph->up;
  //   }
  // }

  // // TODO (0c8406) : Must revise entire logic to better implementation.

  // // In case only one subgraph exists.
  // if (runtime->graph->nodes.size() == 1) {
  //   next_subgraphs_to_invoke.first = runtime->graph->nodes[0]->subgraph_id;
  //   next_subgraphs_to_invoke.second = runtime->graph->nodes[0]->co_subgraph_id;

  //   // case of final subgraph..
  //   if (rx_packet.cur_subgraph == runtime->graph->nodes[0]->subgraph_id) {
  //     next_subgraphs_to_invoke.first = -1;
  //     next_subgraphs_to_invoke.second = -1;
  //   }
  //   return next_subgraphs_to_invoke;
  // }

  // subgraph_node* next_base_subgraph = nullptr;
  // subgraph_node* next_subgraph_to_invoke = nullptr;

  // // case of final subgraph
  // if (prev_base_subgraph->right == nullptr) {
  //   std::cout << "end" << "\n";
  //   next_subgraphs_to_invoke.first = -1;
  //   next_subgraphs_to_invoke.second = -1;
  //   return next_subgraphs_to_invoke;
  // }  // case of first subgraph
  // else if (rx_packet.cur_subgraph == -1) {
  //   next_base_subgraph = root_graph;
  // } else {
  //   next_base_subgraph = prev_base_subgraph->right;
  // }

  // int next_resource_plan = -1;
  // next_subgraph_to_invoke = next_base_subgraph;
  // // ISSUE ,MUST FIX (07b4f) : Consider the gpu utilization ratio delay.
  // // NEED_REFACTOR (02634) : Must change to use obvious resource type.
  // int next_cpu_resource = 0;
  // float gpu_util = monitor->GetGPUUtil();
  // float cpu_util = monitor->GetCPUUtil();

  // /* [IMPT] Utilization based resource allocation part */
  // // std::cout << "CPU : " << cpu_util << " GPU : " << gpu_util << "\n"; 
  
  // // if (gpu_util == 0 && cpu_util == 400) {
  // //   // Use CPU
  // //   next_resource_plan = TF_P_PLAN_GPU;
  // //   std::cout << "USE GPU" << "\n";
  // // } else if (gpu_util == 100 && cpu_util == 0) {
  // //   // Use GPU
  // //   next_resource_plan = TF_P_PLAN_CPU_XNN;
  // //   std::cout << "USE CPU" << "\n";
  // // } else if (gpu_util == 100 && cpu_util == 200) {
  // //   // Use Co-execution
  // //   cpu_usage_flag = true;
  // //   std::cout << "USE CPU200" << "\n";
  // // } else {
  // //   // base plan
  // //   cpu_usage_flag = false;
  // //   std::cout << "USE BASE" << "\n";
  // //   next_resource_plan = next_base_subgraph->resource_type;
  // // }

  // std::cout << "USE BASE" << "\n";
  // next_resource_plan = next_base_subgraph->resource_type;
  
  // /* [IMPT] */

  // // TODO (f85fa) : Fix graph searching, especially in co-execution.
  // // Search for matching subgraph.
  // while (next_subgraph_to_invoke != nullptr) {
  //   if(cpu_usage_flag){
  //     if(next_subgraph_to_invoke->partitioning_ratio == 2)
  //       break;
  //   }else if (next_subgraph_to_invoke->resource_type == next_resource_plan) {
  //     break;
  //   }
  //   if (next_subgraph_to_invoke->down != nullptr) {
  //     next_subgraph_to_invoke = next_subgraph_to_invoke->down;
  //   } else {
  //     next_subgraph_to_invoke = next_base_subgraph;
  //     break;
  //   }
  // }

  // // std::cout << "set next_subgraph_to_invoke id " <<
  // // next_subgraph_to_invoke->subgraph_id << "\n"; std::cout << "set
  // // next_subgraph_to_invoke co id " << next_subgraph_to_invoke->co_subgraph_id
  // // << "\n"; std::cout << "set next_subgraph_to_invoke resource_type " <<
  // // next_subgraph_to_invoke->resource_type << "\n";
  // next_subgraphs_to_invoke.second = next_subgraph_to_invoke->co_subgraph_id;
  // next_subgraphs_to_invoke.first = next_subgraph_to_invoke->subgraph_id;
  // next_subgraphs_to_invoke.second = next_subgraph_to_invoke->co_subgraph_id;

  return next_subgraphs_to_invoke;
}

// [VLS] - maybe deprecated in multi-level design.
// void TfScheduler::PrepareRuntime(tf_packet& rx_packet) {
//   int runtime_id = rx_packet.runtime_id;
//   runtime_* runtime = nullptr;
//   for (int i = 0; i < runtimes.size(); ++i) {
//     if (runtimes[i]->id == runtime_id) runtime = runtimes[i];
//   }
//   // TODO(28caeaf) : Read the subgraph ids from packet and make it as
//   // linked-list?
//   if (runtime == nullptr) {
//     std::cout << "Cannot find matching runtime in PrepareRuntime()"
//               << "\n";
//     exit(-1);
//   }
//   std::queue<int> co_subgraph_ids;
//   std::vector<int> subgraph_ids;
//   int idx = 0;
//   int num_co_subs = 0;

//   while (rx_packet.subgraph_ids[0][idx] != -1) {
//     subgraph_ids.push_back(rx_packet.subgraph_ids[0][idx]);
//     idx++;
//   }
//   idx = 0;
//   while (rx_packet.subgraph_ids[1][idx] != -1) {
//     num_co_subs++;
//     co_subgraph_ids.push(rx_packet.subgraph_ids[1][idx]);
//     idx++;
//   }

//   // Register main subgraphs
//   for (int i = 0; i < subgraph_ids.size(); ++i) {
//     runtime->graph->nodes[i]->subgraph_id = subgraph_ids[i];
//   }
//   // Register Co subgraphs 
//   // MUST FIX(b6582) : co-subgraph does not always exist at the end of whole
//   // graph structure.
//   idx = 0;
//   while (!co_subgraph_ids.empty()) {
//     if (runtime->graph->nodes[idx]->resource_type == 2 ||
//         runtime->graph->nodes[idx]->resource_type == 4) {
//       runtime->graph->nodes[idx]->co_subgraph_id = co_subgraph_ids.front();
//       co_subgraph_ids.pop();
//     }
//     idx++;
//   }

//   if ((subgraph_ids.size() - num_co_subs) != runtime->graph->nodes.size()) {
//     std::cout << "Subgraph ids from runtime and existing graph"
//               << " does not match"
//               << "\n";
//     return;
//   }
// }

// [VLS Todo] call this function multi-level safely.
void TfScheduler::PrepareRuntime(tf_initialization_packet& rx_packet) {
  int runtime_id = rx_packet.runtime_id;
  int level = rx_packet.level;
  runtime_* runtime = nullptr;
  for (int i = 0; i < runtimes.size(); ++i) {
    if (runtimes[i]->id == runtime_id) runtime = runtimes[i];
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
  std::vector<int> subgraph_ids;
  int idx = 0;
  int num_co_subs = 0;

  while (rx_packet.subgraph_ids[0][idx] != -1) {
    subgraph_ids.push_back(rx_packet.subgraph_ids[0][idx]);
    idx++;
  }
  idx = 0;
  while (rx_packet.subgraph_ids[1][idx] != -1) {
    num_co_subs++;
    co_subgraph_ids.push(rx_packet.subgraph_ids[1][idx]);
    idx++;
  }

  // Register main subgraphs
  for (int i = 0; i < subgraph_ids.size(); ++i) {
    runtime->graphs[level]->nodes[i]->subgraph_id = subgraph_ids[i];
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

  if ((subgraph_ids.size() - num_co_subs) != runtime->graphs[level]->nodes.size()) {
    std::cout << "Subgraph ids from runtime and existing graph"
              << " does not match"
              << "\n";
    return;
  }
}

// void TfScheduler::CreateGraphofSubgraphs(tf_packet& tx_packet) {
//   for (int runtime_idx = 0; runtime_idx < runtimes.size(); ++runtime_idx) {
//     if (runtimes[runtime_idx]->id == tx_packet.runtime_id) {
//       runtime_* working_runtime = runtimes[runtime_idx];
//       if (working_runtime->graph != nullptr) {
//         std::cout << "Scheudler: Runtime " << working_runtime->id
//                   << " already has graph."
//                   << "\n";
//         exit(-1);
//       }
//       // Create new graph structure for current runtime.
//       working_runtime->graph = new subgraph_graph;
//       working_runtime->graph->runtime_id = tx_packet.runtime_id;
//       int working_idx = 0;
//       int current_value = tx_packet.partitioning_plan[working_idx]; // Get first value of partitioning plan
//       subgraph_node* new_node;
//       int start_node, end_node, partitioning_ratio, resource_type;
//       bool start_node_flag = true;
//       bool root_graph = true;
//       while(current_value != PART_PARM_SEP_ENDP){  
//         if(current_value == PART_PARM_SEP_OP){ // means end of node subset 
//                                                  // (initialize new subgraph_node)
//           end_node = tx_packet.partitioning_plan[working_idx - 1];
//           start_node_flag = true; // refresh used flag
//           working_idx += 2;
//         }else if(current_value == PART_PARM_SEP_RESROURCE){ // means end of resource & partitioning plan 
//                                                        // (add current subgraph node)
//           partitioning_ratio = tx_packet.partitioning_plan[working_idx - 1];
//           resource_type = tx_packet.partitioning_plan[working_idx - 2];
//           if(root_graph){ // add this graph to root graph.
//             root_graph = false;
//             new_node->rank = 0;
//             new_node->partitioning_ratio = partitioning_ratio;
//             new_node->resource_type = resource_type;
//             new_node->node_start = start_node;
//             new_node->node_end = end_node;
//             working_runtime->graph->root = new_node;
//             working_runtime->graph->nodes.push_back(new_node);
//           }else{ // add this graph to leaf graph.
//             if(!AddSubgraphtoGraph(working_runtime->graph,
//                                    start_node,
//                                    end_node,
//                                    resource_type,
//                                    partitioning_ratio)){
//               std::cout << "AddSubgraphtoGraph ERROR" << "\n";
//               return;
//             }
//           }
//         }else if(current_value != -3){ // means node subset (add)
//           if(start_node_flag){
//             new_node = new subgraph_node;
//             start_node = current_value;
//             start_node_flag = false;
//             if(working_idx == 0) { root_graph = true; }
//           }
//         }
//         working_idx++;
//         current_value = tx_packet.partitioning_plan[working_idx];
//       }
//     }
//   }
//   return;
// }

void TfScheduler::CreateGraphofSubgraphs(int id, 
                                        std::vector<std::vector<int>>& subgraph_params) {
  for (int runtime_idx = 0; runtime_idx < runtimes.size(); ++runtime_idx) {
    if (runtimes[runtime_idx]->id == id) {
      runtime_* working_runtime = runtimes[runtime_idx];
      if (!working_runtime->graphs.empty()) {
        std::cout << "Scheudler: Runtime " << working_runtime->id
                  << " already has graph."
                  << "\n";
        exit(-1);
      }
      for(int level = 0; level<subgraph_params.size(); ++level){ // iterate subgraph-level
        
        // [VLS Todo] change to create subgraph_graph in multi-level.
        // Create new graph structure for current runtime.
        subgraph_graph* new_level_subgraph = new subgraph_graph;
        new_level_subgraph->runtime_id = id;
        new_level_subgraph->level = level;
        
        int working_idx = 0;
        // int current_value = tx_packet.partitioning_plan[working_idx]; // Get first value of partitioning plan
        int current_value = subgraph_params[level][working_idx]; // Get first value of partitioning plan
        subgraph_node* new_node;
        int start_node, end_node, partitioning_ratio, resource_type;
        bool start_node_flag = true;
        bool root_graph = true;
        while(current_value != PART_PARM_SEP_ENDP){  
          if(current_value == PART_PARM_SEP_OP){ // means end of node subset 
                                                  // (initialize new subgraph_node)
            // end_node = tx_packet.partitioning_plan[working_idx - 1];
            end_node = subgraph_params[level][working_idx-1];
            start_node_flag = true; // refresh used flag
            working_idx += 2;
          }else if(current_value == PART_PARM_SEP_RESROURCE){ // means end of resource & partitioning plan 
                                                        // (add current subgraph node)
            // partitioning_ratio = tx_packet.partitioning_plan[working_idx - 1];
            partitioning_ratio = subgraph_params[level][working_idx - 1];
            // resource_type = tx_packet.partitioning_plan[working_idx - 2];
            resource_type = subgraph_params[level][working_idx - 2];
            if(root_graph){ // add this graph to root graph.
              root_graph = false;
              new_node->rank = 0;
              new_node->partitioning_ratio = partitioning_ratio;
              new_node->resource_type = resource_type;
              new_node->node_start = start_node;
              new_node->node_end = end_node;
              new_level_subgraph->root = new_node;
              new_level_subgraph->nodes.push_back(new_node);
            }else{ // add this graph to leaf graph.
              if(!AddSubgraphtoGraph(new_level_subgraph,
                                    start_node,
                                    end_node,
                                    resource_type,
                                    partitioning_ratio)){
                std::cout << "AddSubgraphtoGraph ERROR" << "\n";
                return;
              }
            }
          }else if(current_value != -3){ // means node subset (add)
            if(start_node_flag){
              new_node = new subgraph_node;
              start_node = current_value;
              start_node_flag = false;
              if(working_idx == 0) { root_graph = true; }
            }
          }
          working_idx++;
          // current_value = tx_packet.partitioning_plan[working_idx];
          current_value = subgraph_params[level][working_idx];
        }// subgraph interation
        working_runtime->graphs.push_back(new_level_subgraph);
      }// subgraph level iteration
    } // runtime iteration
  }
  return;
}

bool TfScheduler::AddSubgraphtoGraph(subgraph_graph* graph, int s_node,
                                     int e_node, int resource_type,
                                     int partitioning_ratio) {
  subgraph_node* pointer = graph->root;
  subgraph_node* new_node;
  int new_rank = 0;
  pointer = SearchAndReturnBaseNode(pointer, s_node, e_node, new_rank);

  new_node = new subgraph_node;
  new_node->node_start = s_node;
  new_node->node_end = e_node;
  new_node->resource_type = resource_type;
  new_node->partitioning_ratio = partitioning_ratio;
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
void TfScheduler::CreatePartitioningPlan(tf_initialization_packet& rx_p,
                                         tf_initialization_packet& tx_p) {
  std::vector<std::vector<int>> subgraph_params;
  int layers = 0;
  for (int i = 0; i < 1000; ++i) {
    if (rx_p.latency[i] == -1)
      layers++;
    else
      break;
  }

  // check number of layers
  std::cout << "Runtime [" << rx_p.runtime_id << "] has " << layers
            << " layers in model"
            << "\n";
  for(auto param_file : param_files){ // iterate each parameter files.

  }

  std::string line, token;
  int arg = 0;
  int line_iter = 0; // line reader
  int plan_idx = 0;
  bool seperator_flag = false;
  // get line
  if(!param_file.is_open()){
    std::cout << "Scheduler ERROR : Param file is not opened" << "\n";
    exit(-1);
  }
  // get param line by line
  while (std::getline(param_file, line)) {
    switch (line_iter)
    {
    case 0:{
      if(line == "*"){ // subgraph candidate set end flag
        tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_SUBG;
        plan_idx++;
      }else if(line == "-"){ // subgraph level end flag
        tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_ENDP;
        plan_idx++;
      }else{ // operators
        std::stringstream s_stream(line); // parse line to stringstream.
        while(getline(s_stream, token, ' ')){
          arg = std::stoi(token);  
          tx_p.partitioning_plan[plan_idx] = arg;
          plan_idx++;
          line_iter = 1;
        }  
      }
      break;
    }
    case 1:
      tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_OP;
      plan_idx++;
      tx_p.partitioning_plan[plan_idx] = std::stoi(line);
      plan_idx++;
      line_iter = 2;
      break;
    case 2:
      tx_p.partitioning_plan[plan_idx] = std::stoi(line);
      plan_idx++;
      tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_RESROURCE;
      plan_idx++;
      line_iter = 0;
      break;
    default:
      break;
    }
  }

  // std::cout << "closed" << "\n";
  for(int i=0; i<1000; ++i){
    // std::cout << "adsfasdf" << "\n";
    std::cout << tx_p.partitioning_plan[i] << " ";
    if(tx_p.partitioning_plan[i] == -4)
      break;
  }
  return;
}
// [VLS Todo] change to create multi-level subgraph partitioning plan.
void TfScheduler::CreatePartitioningPlan(tf_initialization_packet& rx_p,
                          std::vector<std::vector<int>>& subgraph_params) {
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
          // tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_SUBG;
          // plan_idx++;
        }else if(line == "-"){ // subgraph level end flag
          subgraph_params[level].push_back(PART_PARM_SEP_SUBG);
          // tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_ENDP;
          // plan_idx++;
        }else{ // operators
          std::stringstream s_stream(line); // parse line to stringstream.
          while(getline(s_stream, token, ' ')){
            arg = std::stoi(token);  
            subgraph_params[level].push_back(arg);
            // tx_p.partitioning_plan[plan_idx] = arg;
            // plan_idx++;
            line_iter = 1;
          }  
        }
        break;
      }
      case 1:
        subgraph_params[level].push_back(PART_PARM_SEP_OP);
        // tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_OP;
        // plan_idx++;
        subgraph_params[level].push_back(std::stoi(line));
        // tx_p.partitioning_plan[plan_idx] = std::stoi(line);
        // plan_idx++;
        line_iter = 2;
        break;
      case 2:
        subgraph_params[level].push_back(std::stoi(line));
        // tx_p.partitioning_plan[plan_idx] = std::stoi(line);
        // plan_idx++;
        subgraph_params[level].push_back(PART_PARM_SEP_RESROURCE);
        // tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_RESROURCE;
        // plan_idx++;
        line_iter = 0;
        break;
      default:
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

TfScheduler::~TfScheduler() {
  std::cout << "Scheduler Terminated"
            << "\n";
};

}  // namespace tflite
