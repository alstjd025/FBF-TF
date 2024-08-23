#pragma once
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
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

  // NEED_REFACTOR (02634) : Add obvious resource type (CPU, GPU, TPU,,.)
  //                         and change resource_type to subgraph_type.
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

    /* partitioning ratio for partitioned subgraph (for co-execution) */
    int partitioning_ratio = -1;

    /* rank of subgraph in 'graph' (need for maintaining the graph struct)*/
    int rank = -1;

    int gpu_utilization_require = 0;   
    int cpu_utilization_require = 0;   
    float average_latency = 0;

    subgraph_node* right =  nullptr;
    subgraph_node* left =  nullptr;
    subgraph_node* up   =  nullptr;
    subgraph_node* down =  nullptr;

  }subgraph_node;

  typedef struct subgraph_graph{
    std::vector<subgraph_node*> nodes;
    subgraph_node* root = nullptr;
    int runtime_id;
    int level;
  }subgraph_graph;

  typedef struct runtime_{
    // will be deprecated by variable length subgraph design below.
    // subgraph_graph* graph = nullptr;
    
    std::vector<subgraph_graph*> graphs; // for multi-level subgraph design.
    // Graphs have multiple level(in vector index).
    // graphs[0] -> subgraphs in level 1 (ex, smallest subgraph granularity)
    // graphs[1] -> subgraphs in level 2 (ex, midium subgraph granularity)
    // ..
    // ..

    int id;
    RuntimeState state;
    struct sockaddr_un addr;
    float latency[TF_P_PLAN_LENGTH];
    int partitioning_plan[TF_P_PLAN_LENGTH];
  }runtime_;

  class TfScheduler{
    public:
      TfScheduler();
      TfScheduler(const char* uds_file_name,
                  std::vector<std::string>& param_file_names);

      void PrintRuntimeStates();

      void Work();

      void OpenPartitioningParams(std::vector<std::string>& param_file_names);

      int SendPacketToRuntime(tf_initialization_packet& tx_p, struct sockaddr_un& runtime_addr);
      int SendPacketToRuntime(tf_runtime_packet& tx_p, struct sockaddr_un& runtime_addr);
      int SendPacketToRuntime(tf_packet& tx_p, struct sockaddr_un& runtime_addr);
      
      int ReceivePacketFromRuntime(tf_initialization_packet& rx_p, struct sockaddr_un& runtime_addr);
      int ReceivePacketFromRuntime(tf_runtime_packet& rx_p, struct sockaddr_un& runtime_addr);
      int ReceivePacketFromRuntime(tf_packet& rx_p, struct sockaddr_un& runtime_addr);
      
      // refresh runtime state in scheduler.
      void RefreshRuntimeState(tf_initialization_packet& rx_p);
      void RefreshRuntimeState(tf_runtime_packet& rx_p);
      void RefreshRuntimeState(tf_packet& rx_p);

      ////////////////////////////////////////////////////////////////////////////////////////
      /* Function description of CreatePartitioningPlan
      Read partitioning parameters from file.
      The parameter array follows the format below.
      format : node_subset /-1/ Resource type(CPU, GPU,,) / Partitioning ratio /-2/ :|(repeat)
               /-3/ new node_subset(redundant subgraph set)/-1/ .. /-2/../-3/.... /-4/(end)
              
              Seperators
              -1 : node subset seperator
              -2 : partitioning, resource plan seperator (also seperates single subgraph)
              -3 : subgraph subset seprator
              -4 : EOF
      The parameter FILE follows the format below.
      ex) sample_param_file
      0 1 2 3 6 7               ----- node subset 1
      3                         ----- resource plan (CPU, GPU, CO-GPU, XNN,,)
      15                        ----- partitioning ratio for co-execution (height, channel)
      4 5                       ----- node subset 2
      0                         ----- resource plan (CPU, GPU, CO-GPU, XNN,,)
      0                         ----- partitioning ratio for co-execution (height, channel)
      *                         ----- subset seperator
      0 1 2 3 6 7               ----- node subset 1-1 (same node indices with different resource)
      1                         ----- resource plan (CPU, GPU, CO-GPU, XNN,,)
      15                        ----- partitioning ratio for co-execution (height, channel)
      *                         ----- subset seperator
      -                         ----- EOF
      file will be parsed as array below
      ary = [0 1 2 3 6 7 -1 0 0 -2 4 5 -1 0 0 -2 -3 0 1 2 3 6 7 -1 4 15 -2 -3 -4]
      which means,
      subgraph 1 : 0,1,2,3,6,7
       - r_type  : 0 (CPU)
       - p_ratio : 0 (no partitioning)
      subgraph 2 : 4,5
       - r_type  : 0 (CPU)
       - p_ratio : 0 (no partitioning)
      subgraph 1-2 : 0,1,2,3,6,7
       - r_type  : 4 (CO-XNN) 
       - p_ratio : 15 (5:5) */
       ////////////////////////////////////////////////////////////////////////////////////////
      
      void CreatePartitioningPlan(tf_initialization_packet& rx_p, 
                                  std::vector<std::vector<int>>& subgraph_param);

      // Create a graph of subgraphs.
      // void CreateGraphofSubgraphs(tf_packet& tx_packet); // deprecated [VLS] - for multi-level subgraph.
      void CreateGraphofSubgraphs(int id, std::vector<std::vector<int>>& subgraph_param);

      // Add new graph node to graph.
      bool AddSubgraphtoGraph(subgraph_graph* graph, int s_node, int e_node,
                              int resource_type, int partitioning_ratio);

      // Graph search functions.
      // Search the graph structure down to up & left to right.
      // This function returns pointer of 'base node'....
      subgraph_node* SearchAndReturnBaseNode(subgraph_node* node, int s_node,
                                                    int e_node, int& new_rank);
      
      // Search and return pointer of subgraph_node with given id and root. 
      subgraph_node* SearchAndReturnNodeWithID(subgraph_node* root, int id);

      void PrintGraph(int runtime_id);

      // Search and return the subgraph's id to invoke.    
      std::pair<int, int> SearchNextSubgraphtoInvoke(tf_runtime_packet& rx_packet);

      // Refresh the whole graph structure of current runtime and finally add
      // 'id' in them.
      // void PrepareRuntime(tf_packet& rx_packet); // deprecated [VLS] - for multi-level subgraph.
      void PrepareRuntime(tf_initialization_packet& rx_packet);

      bool CheckAllRuntimesReady();

      void ReleaseResource(ResourceType type);

      ~TfScheduler();
    
    private:
      LiteSysMonitor* monitor;
      std::vector<std::fstream*> param_files;
      std::fstream param_file; // delete after variable length subgraph impl.
      std::vector<std::vector<int>> subgraph_params; // experimental

      int scheduler_fd;
      size_t addr_size;
      struct sockaddr_un scheduler_addr;

      int first_counter = 0;
      int second_counter = 0;
      int third_counter = 0;
      int fourth_counter = 0;
      int fifth_counter = 0;

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
      float* gpu_util;
      
      // current CPU utlization ratio(average).
      float* cpu_util;
  };

}