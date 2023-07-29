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
#include "tensorflow/lite/tf_monitor.h"

namespace tflite{


  typedef struct subgraph_node{
    /* id of subgraph */
    int subgraph_id = -1;      
    
    /* id of co-execution subgraph (if exists) */
    int co_subgraph_id = -1;   
    
    /* start node of subgraph */
    int node_start = -1;       
    
    /* end node of subgraph */
    int node_end = -1;         
    
    /* resource type of subgraph (CPU, GPU, CO-EX) */
    int resource_type = -1;    

    /* rank of subgraph in 'graph' (need for maintaining the graph struct)*/
    int rank = -1;   

    subgraph_node* right =  nullptr;
    subgraph_node* left =  nullptr;
    subgraph_node* up   =  nullptr;
    subgraph_node* down =  nullptr;

  }subgraph_node;

  typedef struct subgraph_graph{
    std::vector<subgraph_node*> nodes;
    subgraph_node* root = nullptr;
    int runtime_id;
  }subgraph_graph;

  typedef struct runtime_{
    subgraph_graph* graph = nullptr;
    int id;
    RuntimeState state;
    struct sockaddr_un addr;
    float latency[TF_P_PLAN_LENGTH];
    int partitioning_plan[TF_P_PLAN_LENGTH][TF_P_PLAN_SIZE];
    // First idx means first subgraph's idx in partitioning subset.
    // Second idx means last subgraph's idx in subset.
    // Third idx means processor to be used for invoke.
    //  0 - cpu, 1 - gpu, 2 - co_execution flag
    // Fourth idx means partitioning ratio(1~19).
    //  3 means, GPU : 3  CPU : 7 (for channel-wise)
    // 13 means, GPU : 3  CPU : 7 (for height-wise)
  }runtime_;

  class TfScheduler{
    public:
      TfScheduler();
      TfScheduler(const char* uds_file_name);

      void PrintRuntimeStates();

      void Work();

      void SysMonitor();

      int SendPacketToRuntime(tf_packet& tx_p, struct sockaddr_un& runtime_addr);
      
      int ReceivePacketFromRuntime(tf_packet& rx_p, struct sockaddr_un& runtime_addr);
      
      // refresh runtime state in scheduler.
      void RefreshRuntimeState(tf_packet& rx_p);

      void CreatePartitioningPlan(tf_packet& rx_p, tf_packet& tx_p);

      // Create a graph of subgraphs.
      void CreateGraphofSubgraphs(tf_packet& tx_packet);

      // Add new graph node to graph.
      bool AddSubgraphtoGraph(subgraph_graph* graph, int s_node, int e_node,
                              int resource_type);

      // Graph search functions.
      subgraph_node* SearchAndReturnBaseNode(subgraph_node* node, int s_node,
                                                    int e_node, int& new_rank);
      
      void PrintGraph(int runtime_id);

      // Search and return the subgraph's id to invoke.    
      std::pair<int, int>& SearchNextSubgraphtoInvoke(int runtime_id);

      // Refresh the whole graph structure of current runtime and finally add
      // 'id' in them.
      void PrepareRuntime(tf_packet& rx_packet);

      bool CheckAllRuntimesReady();

      bool RoundRobin(ResourceType type, int runtime_id);
      void ReleaseResource(ResourceType type);

      ~TfScheduler();
    
    private:

    LiteSysMonitor* monitor;

    int scheduler_fd;
    size_t addr_size;
    struct sockaddr_un scheduler_addr;

    std::vector<runtime_*> runtimes;
    int runtimes_created = 0;

    bool reschedule_needed = false;

    // For RR scheduler
    bool cpu_usage_flag = false;
    bool gpu_usage_flag = false;
    bool cpgpu_usage_flag = false;
    std::queue<int> rr_cpu_queue;
    std::queue<int> rr_gpu_queue;

    // current GPU utlization ratio.
    float gpu_util;
    
    // current CPU utlization ratio(average).
    float cpu_util;
  };

}