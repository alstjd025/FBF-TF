#include "tensorflow/lite/tf_scheduler.h"

namespace tflite {

TfScheduler::TfScheduler(){};

TfScheduler::TfScheduler(const char* uds_file_name,
                         const char* partitioning_params) {
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
  param_file_name = partitioning_params;
  OpenPartitioningParams();

  std::cout << "Scheduler initializaing done"
            << "\n";
};

int TfScheduler::SendPacketToRuntime(tf_packet& tx_p,
                                     struct sockaddr_un& runtime_addr) {
  int v;
  v = sendto(scheduler_fd, (void*)&tx_p, sizeof(tf_packet), 0,
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

void TfScheduler::SysMonitor() {
  std::cout << "[Scheduler] System monitoring started"
            << "\n";
  // std::thread cpu_monitor_daemon, gpu_monitor_daemon;
  // cpu_monitor_daemon = std::thread(&GetCPUUtilization, cpu_util);
  // gpu_monitor_daemon = std::thread(&GetGPUUtilization, gpu_util);
  // cpu_monitor_daemon.join();
  // gpu_monitor_daemon.join();
}

void TfScheduler::Work() {
  monitor = new LiteSysMonitor();
  bool run = true;
  while (run) {
    tf_packet rx_packet;
    struct sockaddr_un runtime_addr;
    memset(&rx_packet, 0, sizeof(tf_packet));
    if (ReceivePacketFromRuntime(rx_packet, runtime_addr) == -1) {
      std::cout << "Receive failed"
                << "\n";
      return;
    }
    // std::cout << "Recieved packet from runtime " << rx_packet.runtime_id <<
    // "\n";

    // do next work by received runtime state.
    switch (rx_packet.runtime_current_state) {
      case RuntimeState::INITIALIZE: {
        std::cout << "runtime init"
                  << "\n";
        for (auto runtime : runtimes) {
          if (runtime->id == rx_packet.runtime_id) {
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

        tf_packet tx_packet;
        memset(&tx_packet, 0, sizeof(tf_packet));
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
        tf_packet tx_packet;
        RefreshRuntimeState(rx_packet);
        CreatePartitioningPlan(rx_packet, tx_packet);

        tx_packet.runtime_id = rx_packet.runtime_id;
        tx_packet.runtime_next_state = RuntimeState::SUBGRAPH_CREATE;

        // Not done
        std::cout << "CreateGraphofSubgraphs" << "\n";
        CreateGraphofSubgraphs(tx_packet);
        std::cout << "CreateGraphofSubgraphs done" << "\n";
        if (SendPacketToRuntime(tx_packet, runtime_addr) == -1) {
          std::cout << "sock : " << runtime_addr.sun_path << " "
                    << runtime_addr.sun_family << "\n";
          printf("errno : %d \n", errno);
          return;
        }
        break;
      }
      case RuntimeState::SUBGRAPH_CREATE: {
        RefreshRuntimeState(rx_packet);

        tf_packet tx_packet;
        tx_packet.runtime_id = rx_packet.runtime_id;
        tx_packet.runtime_next_state = RuntimeState::INVOKE_;

        // not done
        std::cout << "Prepare runtime " << "\n";
        PrepareRuntime(rx_packet);
        std::cout << "Prepare runtime done" << "\n";
        PrintGraph(rx_packet.runtime_id);

        if (SendPacketToRuntime(tx_packet, runtime_addr) == -1) {
          std::cout << "sock : " << runtime_addr.sun_path << " "
                    << runtime_addr.sun_family << "\n";
          printf("errno : %d \n", errno);
          return;
        }
        break;
      }
      case RuntimeState::INVOKE_: {
        // std::cout << "runtime invoke" << "\n";
        RefreshRuntimeState(rx_packet);
        tf_packet tx_packet;
        tx_packet.runtime_id = rx_packet.runtime_id;
        tx_packet.runtime_next_state = RuntimeState::INVOKE_;

        std::pair<int, int> next_subgraph_to_invoke;
        next_subgraph_to_invoke = SearchNextSubgraphtoInvoke(rx_packet);
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

void TfScheduler::OpenPartitioningParams() {
  param_file.open(param_file_name, std::fstream::in);
}

std::pair<int, int> TfScheduler::SearchNextSubgraphtoInvoke(
    tf_packet& rx_packet) {
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

  int gpu_thresh = 70;
  int cpu_thresh = 70;
  subgraph_node* root_graph = runtime->graph->root;
  subgraph_node* prev_invoked_subgraph = nullptr;
  subgraph_node* prev_base_subgraph = nullptr;
  if (rx_packet.cur_subgraph == -1) {  // first invoke
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

  // TODO (0c8406) : Must revise entire logic to better implementation.

  // In case only one subgraph exists.
  if (runtime->graph->nodes.size() == 1) {
    next_subgraphs_to_invoke.first = runtime->graph->nodes[0]->subgraph_id;
    next_subgraphs_to_invoke.second = runtime->graph->nodes[0]->co_subgraph_id;

    // case of final subgraph..
    if (rx_packet.cur_subgraph == runtime->graph->nodes[0]->subgraph_id) {
      next_subgraphs_to_invoke.first = -1;
      next_subgraphs_to_invoke.second = -1;
    }
    return next_subgraphs_to_invoke;
  }

  subgraph_node* next_base_subgraph = nullptr;
  subgraph_node* next_subgraph_to_invoke = nullptr;

  // case of final subgraph
  if (prev_base_subgraph->right == nullptr) {
    next_subgraphs_to_invoke.first = -1;
    next_subgraphs_to_invoke.second = -1;
    return next_subgraphs_to_invoke;
  }  // case of first subgraph
  else if (rx_packet.cur_subgraph == -1) {
    next_base_subgraph = root_graph;
  } else {
    next_base_subgraph = prev_base_subgraph->right;
  }

  int next_resource_plan = -1;
  next_subgraph_to_invoke = next_base_subgraph;
  // ISSUE ,MUST FIX (07b4f) : Consider the gpu utilization ratio delay.
  // NEED_REFACTOR (02634) : Must change to use obvious resource type.
  float gpu_util = monitor->GetGPUUtil();
  float cpu_util = monitor->GetCPUUtil();
  if (gpu_util > gpu_thresh && cpu_util < cpu_thresh) {
    // Use CPU
    next_resource_plan = TF_P_PLAN_CPU;
  } else if (gpu_util < gpu_thresh && cpu_util > cpu_thresh) {
    // Use GPU
    next_resource_plan = TF_P_PLAN_GPU;
  } else if (gpu_util > gpu_thresh && cpu_util > cpu_thresh) {
    // Use Co-execution
    next_resource_plan = TF_P_PLAN_CO_E;
  } else {
    // base plan
    next_resource_plan = next_base_subgraph->resource_type;
  }

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

void TfScheduler::PrepareRuntime(tf_packet& rx_packet) {
  int runtime_id = rx_packet.runtime_id;
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
    runtime->graph->nodes[i]->subgraph_id = subgraph_ids[i];
  }
  // Register Co subgraphs
  // MUST FIX(b6582) : co-subgraph does not always exist at the end of whole
  // graph structure.
  idx = 0;
  while (!co_subgraph_ids.empty()) {
    if (runtime->graph->nodes[idx]->resource_type == 2 ||
        runtime->graph->nodes[idx]->resource_type == 4) {
      runtime->graph->nodes[idx]->co_subgraph_id = co_subgraph_ids.front();
      co_subgraph_ids.pop();
    }
    idx++;
  }

  if ((subgraph_ids.size() - num_co_subs) != runtime->graph->nodes.size()) {
    std::cout << "Subgraph ids from runtime and existing graph"
              << " does not match"
              << "\n";
    return;
  }
}

void TfScheduler::CreateGraphofSubgraphs(tf_packet& tx_packet) {
  for (int runtime_idx = 0; runtime_idx < runtimes.size(); ++runtime_idx) {
    if (runtimes[runtime_idx]->id == tx_packet.runtime_id) {
      runtime_* working_runtime = runtimes[runtime_idx];
      if (working_runtime->graph != nullptr) {
        std::cout << "Scheudler: Runtime " << working_runtime->id
                  << " already has graph."
                  << "\n";
        exit(-1);
      }
      // Create new graph structure for current runtime.
      working_runtime->graph = new subgraph_graph;
      working_runtime->graph->runtime_id = tx_packet.runtime_id;
      int working_idx = 0;
      int current_value = tx_packet.partitioning_plan[working_idx]; // Get first value of partitioning plan
      subgraph_node* new_node;
      int start_node, end_node, partitioning_ratio, resource_type;
      bool start_node_flag = true;
      bool root_graph = true;
      while(current_value != PART_PARM_SEP_ENDP){  
        std::cout << "current_value : " << current_value << "\n";
        if(current_value == PART_PARM_SEP_NODE){ // means end of node subset 
                                                 // (initialize new subgraph_node)
          end_node = tx_packet.partitioning_plan[working_idx - 1];
          std::cout << "end node " << end_node << "\n"; 
          start_node_flag = true; // refresh used flag
          working_idx += 2;
        }else if(current_value == PART_PARM_SEP_RESR){ // means end of resource & partitioning plan 
                                                       // (add current subgraph node)
          partitioning_ratio = tx_packet.partitioning_plan[working_idx - 1];
          std::cout << "p_ratio " << partitioning_ratio << "\n";
          resource_type = tx_packet.partitioning_plan[working_idx - 2];
          std::cout << "r_type " << resource_type << "\n";
          if(root_graph){ // add this graph to root graph.
            std::cout << "add root graph" << "\n";
            root_graph = false;
            new_node->rank = 0;
            new_node->partitioning_ratio = partitioning_ratio;
            new_node->resource_type = resource_type;
            new_node->node_start = start_node;
            new_node->node_end = end_node;
            working_runtime->graph->root = new_node;
            working_runtime->graph->nodes.push_back(new_node);
          }else{ // add this graph to leaf graph.
            std::cout << "add graph" << "\n";
            if(!AddSubgraphtoGraph(working_runtime->graph,
                                   start_node,
                                   end_node,
                                   resource_type,
                                   partitioning_ratio)){
              std::cout << "AddSubgraphtoGraph ERROR" << "\n";
              return;
            }
            std::cout << "add done" << "\n";
          }
        }else if(current_value != -3){ // means node subset (add)
          if(start_node_flag){
            new_node = new subgraph_node;
            start_node = current_value;
            std::cout << "start node " << current_value << "\n";
            start_node_flag = false;
            if(working_idx == 0) { root_graph = true; }
          }
        }
        working_idx++;
        current_value = tx_packet.partitioning_plan[working_idx];
      }
      std::cout << "adsf" << "\n";
    }
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

  std::cout << "Prints subgraphs in runtime " << runtime->graph->runtime_id
            << "\n";
  for (int i = 0; i < runtime->graph->nodes.size(); ++i) {
    std::cout << "Main subgraph ID " << runtime->graph->nodes[i]->subgraph_id
              << " ";
    std::cout << "Co subgraph ID " << runtime->graph->nodes[i]->co_subgraph_id
              << " ";
    std::cout << "RANK " << runtime->graph->nodes[i]->rank << "\n";
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

bool TfScheduler::RoundRobin(ResourceType type, int runtime_id) {
  // if(runtimes.size() < 2){
  //   return true;
  // }
  if (!CheckAllRuntimesReady()) {  // Every runtime should be in invoke state to
                                   // start RR scheduling.
    return false;
  }
  switch (type) {
    case ResourceType::CPU:
      if (rr_cpu_queue
              .empty()) {  // initial state. any runtime can take ownership
        rr_cpu_queue.push(runtime_id);
        return true;
      } else if (rr_cpu_queue.front() !=
                 runtime_id) {  // if last owner was other runtime
        if (cpu_usage_flag)     // Resource busy.
          return false;
        else {  // Resource available
          rr_cpu_queue.pop();
          rr_cpu_queue.push(runtime_id);
          cpu_usage_flag = true;
          return true;
        }
      } else
        return false;  // if last owner was this runtime
    case ResourceType::GPU:
      if (rr_gpu_queue
              .empty()) {  // initial state. any runtime can take ownership
        rr_gpu_queue.push(runtime_id);
        return true;
      } else if (rr_gpu_queue.front() !=
                 runtime_id) {  // if last owner was other runtime
        if (gpu_usage_flag)     // Resource busy.
          return false;
        else {  // Resource available
          rr_gpu_queue.pop();
          rr_gpu_queue.push(runtime_id);
          cpu_usage_flag = true;
          return true;
        }
      } else
        return false;  // if last owner was this runtime
    // case ResourceType::CPUGPU:
    //   /* Not implemented */
    //   break;
    default:
      break;
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

void TfScheduler::CreatePartitioningPlan(tf_packet& rx_p, tf_packet& tx_p) {
  int layers = 0;
  for (int i = 0; i < 1000; ++i) {
    if (rx_p.latency[i] == -1)
      layers++;
    else
      break;
  }
  std::cout << "Runtime [" << rx_p.runtime_id << "] has " << layers
            << " layers in model"
            << "\n";
  std::string line;
  int arg = 0;
  int line_iter = 0;
  int plan_idx = 0;
  bool seperator_flag = false;
  // get line
  if(!param_file.is_open()){
    std::cout << "Scheduler ERROR : Param file is not opened" << "\n";
    exit(-1);
  }
  while (std::getline(param_file, line)) {
    std::cout << line << "\n";
    switch (line_iter)
    {
    case 0:
      for (int string_idx = 0; string_idx < line.length(); string_idx++) {          
        if(line[string_idx] == '*'){ // subgraph set end flag
          tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_SUBG;
          plan_idx++;
        }else if(line[string_idx] == '-'){ // master end falg
          tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_ENDP;
          plan_idx++;
        }else if(line[string_idx] == ' '){ // node seperator
          continue;
        }else{ // nodes
          arg = std::stoi(&line[string_idx]);  
          tx_p.partitioning_plan[plan_idx] = arg;
          plan_idx++;
          line_iter = 1;
        }
      }
      break;
    case 1:
      tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_NODE;
      plan_idx++;
      tx_p.partitioning_plan[plan_idx] = std::stoi(line);
      plan_idx++;
      line_iter = 2;
      break;
    case 2:
      tx_p.partitioning_plan[plan_idx] = std::stoi(line);
      plan_idx++;
      tx_p.partitioning_plan[plan_idx] = PART_PARM_SEP_RESR;
      plan_idx++;
      line_iter = 0;
      break;
    default:
      break;
    }
  }

  // std::cout << "closed" << "\n";
  // for(int i=0; i<1000; ++i){
  //   // std::cout << "adsfasdf" << "\n";
  //   std::cout << tx_p.partitioning_plan[i] << " ";
  //   if(tx_p.partitioning_plan[i] == -4)
  //     break;
  // }
  return;
  // if(layers == 9){ // MNIST

  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 9;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // if want two subgraph
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 1;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 2; // partitioning ratio
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = 1;
  //   tx_p.partitioning_plan[1][TF_P_IDX_END]      = 9;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[2][TF_P_IDX_START]    = TF_P_END_PLAN;
  // } // MNIST
  // else if(layers == 124){ // MOBILENET_V3 224
  // //(old, from TF model hub)
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 124;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;
  // }else if(layers == 123){ // MOBILENET_V3 224
  // //(from
  // https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 123;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;
  // }else if(layers == 31){ // MOBILENET_V1 224
  // //(from
  // https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/default/1)
  //   // TEST PLAN  -- HW & CH
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 27;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 18; // partitioning ratio
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = 27;
  //   tx_p.partitioning_plan[1][TF_P_IDX_END]      = 29;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 8; // partitioning ratio
  //   tx_p.partitioning_plan[2][TF_P_IDX_START]    = 29;
  //   tx_p.partitioning_plan[2][TF_P_IDX_END]      = 31;
  //   tx_p.partitioning_plan[2][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[2][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[3][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // BASELINE for CPU
  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 31;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // BASELINE for GPU
  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 29;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = 29;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_END]      = 31;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[2][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // TEST PLAN (Xavier) 7.2
  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 29;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 18; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = 29;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_END]      = 31;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[2][TF_P_IDX_START]    = TF_P_END_PLAN;

  // }else if(layers == 118){ // efficientnet lite 4
  // // layers == 118 for GPU FP32
  // // layers == 120 for CPU UINT8
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 114;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 18; // partitioning ratio
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = 114;
  //   tx_p.partitioning_plan[1][TF_P_IDX_END]      = 118;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[2][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 118;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;
  // }else if(layers == 152){ // yolo_v4_tiny-ieie
  //   // baselines
  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 152;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;
  //   // tx_p.partitioning_plan[2][TF_P_IDX_START]    = TF_P_END_MASTER;

  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 8;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = 8;
  //   tx_p.partitioning_plan[1][TF_P_IDX_END]      = 9;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[2][TF_P_IDX_START]    = 9;
  //   tx_p.partitioning_plan[2][TF_P_IDX_END]      = 20;
  //   tx_p.partitioning_plan[2][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   tx_p.partitioning_plan[2][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[3][TF_P_IDX_START]    = 20;
  //   tx_p.partitioning_plan[3][TF_P_IDX_END]      = 21;
  //   tx_p.partitioning_plan[3][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[3][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[4][TF_P_IDX_START]    = 21;
  //   tx_p.partitioning_plan[4][TF_P_IDX_END]      = 32;
  //   tx_p.partitioning_plan[4][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   tx_p.partitioning_plan[4][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[5][TF_P_IDX_START]    = 32;
  //   tx_p.partitioning_plan[5][TF_P_IDX_END]      = 33;
  //   tx_p.partitioning_plan[5][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[5][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[6][TF_P_IDX_START]    = 33; // problem on node 52
  //   tx_p.partitioning_plan[6][TF_P_IDX_END]      = 55; // 102?
  //   tx_p.partitioning_plan[6][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU_XNN;
  //   tx_p.partitioning_plan[6][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   17 tx_p.partitioning_plan[7][TF_P_IDX_START]    = 55;
  //   tx_p.partitioning_plan[7][TF_P_IDX_END]      = 152;
  //   tx_p.partitioning_plan[7][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[7][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[8][TF_P_IDX_START]    = TF_P_END_PLAN;
  //   tx_p.partitioning_plan[9][TF_P_IDX_START]    = TF_P_END_MASTER;

  //   // tx_p.partitioning_plan[9][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[9][TF_P_IDX_END]      = 8;
  //   // tx_p.partitioning_plan[9][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU_XNN;
  //   // tx_p.partitioning_plan[9][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // // tx_p.partitioning_plan[9][TF_P_IDX_START]    = 55;
  //   // // tx_p.partitioning_plan[9][TF_P_IDX_END]      = 152;
  //   // // tx_p.partitioning_plan[9][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // // tx_p.partitioning_plan[9][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // // tx_p.partitioning_plan[10][TF_P_IDX_START]    = TF_P_END_PLAN;
  //   // // tx_p.partitioning_plan[11][TF_P_IDX_START]    = TF_P_END_MASTER;

  //   // tx_p.partitioning_plan[10][TF_P_IDX_START]    = 9;
  //   // tx_p.partitioning_plan[10][TF_P_IDX_END]      = 20;
  //   // tx_p.partitioning_plan[10][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU_XNN;
  //   // tx_p.partitioning_plan[10][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio

  //   // tx_p.partitioning_plan[11][TF_P_IDX_START]    = 21;
  //   // tx_p.partitioning_plan[11][TF_P_IDX_END]      = 32;
  //   // tx_p.partitioning_plan[11][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU_XNN;
  //   // tx_p.partitioning_plan[11][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[12][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // // tx_p.partitioning_plan[12][TF_P_IDX_START]    = 33;
  //   // // tx_p.partitioning_plan[12][TF_P_IDX_END]      = 55;
  //   // // tx_p.partitioning_plan[12][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU_XNN;
  //   // // tx_p.partitioning_plan[12][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // // tx_p.partitioning_plan[13][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // tx_p.partitioning_plan[13][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[13][TF_P_IDX_END]      = 8;
  //   // tx_p.partitioning_plan[13][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E_XNN;
  //   // tx_p.partitioning_plan[13][TF_P_IDX_RATIO]    = 17; // partitioning
  //   ratio

  //   // tx_p.partitioning_plan[14][TF_P_IDX_START]    = 9;
  //   // tx_p.partitioning_plan[14][TF_P_IDX_END]      = 20;
  //   // tx_p.partitioning_plan[14][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E_XNN;
  //   // tx_p.partitioning_plan[14][TF_P_IDX_RATIO]    = 17; // partitioning
  //   ratio

  //   // tx_p.partitioning_plan[15][TF_P_IDX_START]    = 21;
  //   // tx_p.partitioning_plan[15][TF_P_IDX_END]      = 32;
  //   // tx_p.partitioning_plan[15][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E_XNN;
  //   // tx_p.partitioning_plan[15][TF_P_IDX_RATIO]    = 16; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[16][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // tx_p.partitioning_plan[17][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[17][TF_P_IDX_END]      = 8;
  //   // tx_p.partitioning_plan[17][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   // tx_p.partitioning_plan[17][TF_P_IDX_RATIO]    = 17; // partitioning
  //   ratio

  //   // tx_p.partitioning_plan[18][TF_P_IDX_START]    = 9;
  //   // tx_p.partitioning_plan[18][TF_P_IDX_END]      = 20;
  //   // tx_p.partitioning_plan[18][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   // tx_p.partitioning_plan[18][TF_P_IDX_RATIO]    = 17; // partitioning
  //   ratio

  //   // tx_p.partitioning_plan[19][TF_P_IDX_START]    = 21;
  //   // tx_p.partitioning_plan[19][TF_P_IDX_END]      = 32;
  //   // tx_p.partitioning_plan[19][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   // tx_p.partitioning_plan[19][TF_P_IDX_RATIO]    = 16; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[20][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // // tx_p.partitioning_plan[23][TF_P_IDX_START]    = 33;
  //   // // tx_p.partitioning_plan[23][TF_P_IDX_END]      = 55;
  //   // // tx_p.partitioning_plan[23][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   // // tx_p.partitioning_plan[23][TF_P_IDX_RATIO]    = 15; // partitioning
  //   ratio
  //   // // tx_p.partitioning_plan[24][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // tx_p.partitioning_plan[21][TF_P_IDX_START]    = TF_P_END_MASTER;

  //   //
  // }
  // else if(layers == 59){ //yolov4_tiny from pinto
  // // for gpu
  // // 0 ~ 7
  // // 9 ~ 19
  // // 21 ~ 31
  // // 33 ~ 50  -> testing subgraph
  // // for cpu(minimal precision, int8) 38 ~ 57 is co-execution subgraph
  // // node 33 -> conv2d input 1 26 26 128
  // //
  // // 55 ~ 58

  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 8;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = 8;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_END]      = 9;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[2][TF_P_IDX_START]    = 9;
  //   // tx_p.partitioning_plan[2][TF_P_IDX_END]      = 19;
  //   // tx_p.partitioning_plan[2][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   // tx_p.partitioning_plan[2][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[3][TF_P_IDX_START]    = 19;
  //   // tx_p.partitioning_plan[3][TF_P_IDX_END]      = 21;
  //   // tx_p.partitioning_plan[3][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[3][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[4][TF_P_IDX_START]    = 21;
  //   // tx_p.partitioning_plan[4][TF_P_IDX_END]      = 31;
  //   // tx_p.partitioning_plan[4][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   // tx_p.partitioning_plan[4][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[5][TF_P_IDX_START]    = 31;
  //   // tx_p.partitioning_plan[5][TF_P_IDX_END]      = 33;
  //   // tx_p.partitioning_plan[5][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[5][TF_P_IDX_RATIO]    = 0; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[6][TF_P_IDX_START]    = 33;
  //   // tx_p.partitioning_plan[6][TF_P_IDX_END]      = 50;
  //   // tx_p.partitioning_plan[6][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   // tx_p.partitioning_plan[6][TF_P_IDX_RATIO]    = 15; // partitioning
  //   ratio
  //   // tx_p.partitioning_plan[7][TF_P_IDX_START]    = 50;
  //   // tx_p.partitioning_plan[7][TF_P_IDX_END]      = 56;
  //   // tx_p.partitioning_plan[7][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[7][TF_P_IDX_RATIO]    = 0;
  //   // tx_p.partitioning_plan[8][TF_P_IDX_START]    = 56;
  //   // tx_p.partitioning_plan[8][TF_P_IDX_END]      = 59;
  //   // tx_p.partitioning_plan[8][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   // tx_p.partitioning_plan[8][TF_P_IDX_RATIO]    = 0;
  //   // tx_p.partitioning_plan[9][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // CPU execution for debugging
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 59;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0;
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;

  // }
  // else if(layers == 68){ // case of yolo v4 tiny cpu (including quantize
  // layer)
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 8;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = 8;
  //   tx_p.partitioning_plan[1][TF_P_IDX_END]      = 9;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[2][TF_P_IDX_START]    = 9;
  //   tx_p.partitioning_plan[2][TF_P_IDX_END]      = 21;
  //   tx_p.partitioning_plan[2][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[2][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[3][TF_P_IDX_START]    = 21;
  //   tx_p.partitioning_plan[3][TF_P_IDX_END]      = 23;
  //   tx_p.partitioning_plan[3][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[3][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[4][TF_P_IDX_START]    = 23;
  //   tx_p.partitioning_plan[4][TF_P_IDX_END]      = 36;
  //   tx_p.partitioning_plan[4][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[4][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[5][TF_P_IDX_START]    = 36;
  //   tx_p.partitioning_plan[5][TF_P_IDX_END]      = 38;
  //   tx_p.partitioning_plan[5][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[5][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[6][TF_P_IDX_START]    = 38;
  //   tx_p.partitioning_plan[6][TF_P_IDX_END]      = 58;
  //   tx_p.partitioning_plan[6][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[6][TF_P_IDX_RATIO]    = 0; // partitioning ratio
  //   tx_p.partitioning_plan[7][TF_P_IDX_START]    = 58;
  //   tx_p.partitioning_plan[7][TF_P_IDX_END]      = 65;
  //   tx_p.partitioning_plan[7][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[7][TF_P_IDX_RATIO]    = 0;
  //   tx_p.partitioning_plan[8][TF_P_IDX_START]    = 65;
  //   tx_p.partitioning_plan[8][TF_P_IDX_END]      = 68;
  //   tx_p.partitioning_plan[8][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[8][TF_P_IDX_RATIO]    = 0;
  //   tx_p.partitioning_plan[9][TF_P_IDX_START]    = TF_P_END_PLAN;
  // }else if(layers == 52){ // ultra fast lanenet
  // // 52 for FP32. 54 for int8
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 47;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 15;
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = 47;
  //   tx_p.partitioning_plan[1][TF_P_IDX_END]      = 52;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0;
  //   tx_p.partitioning_plan[2][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 47;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_GPU;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = 47;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_END]      = 52;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0;
  //   // tx_p.partitioning_plan[2][TF_P_IDX_START]    = TF_P_END_PLAN;

  //   // FOR INT8
  //   // tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_END]      = 54;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   // tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0;
  //   // tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;

  // }else if(layers == 54){
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 47;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CO_E;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 15;
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = 47;
  //   tx_p.partitioning_plan[1][TF_P_IDX_END]      = 52;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[1][TF_P_IDX_RATIO]    = 0;
  //   tx_p.partitioning_plan[2][TF_P_IDX_START]    = TF_P_END_PLAN;
  // }
  // else{
  //   tx_p.partitioning_plan[0][TF_P_IDX_START]    = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_END]      = 0;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RESOURCE] = TF_P_PLAN_CPU;
  //   tx_p.partitioning_plan[0][TF_P_IDX_RATIO]    = 0;
  //   tx_p.partitioning_plan[1][TF_P_IDX_START]    = TF_P_END_PLAN;
  // }
}

TfScheduler::~TfScheduler() {
  std::cout << "Scheduler Terminated"
            << "\n";
};

}  // namespace tflite
