#include <vector>
#include <string>
#include "tensorflow/lite/c/common.h"

#define KMCONTEXT() kmcontext.setContext(&context_, &execution_plan_, &nodes_and_registration_)

class KmContext {
  public:
    void printNodeIndex();
    void printNodeDims();
	void printOutputTensors();
    void channelPartitioning(std::vector<std::pair<int, float>>& layer);
    void channelPartitioning(std::string op_name, float ratio);
    void channelPartitioning(std::vector<int>& partitioning_plan, std::vector<float>& ratios);

    TfLiteContext* context_;
    std::vector<int>* execution_plan_;
    std::vector<std::pair<TfLiteNode, TfLiteRegistration>>* nodes_and_registration_;

	std::vector<int> partitioning_plan_;
	std::vector<float> ratios_;

    void setContext(TfLiteContext* context, std::vector<int>* execution_plan, 
                    std::vector<std::pair<TfLiteNode, TfLiteRegistration>>* nodes_and_registration);
};

const char* GetOpName(const TfLiteRegistration& op_reg);

extern KmContext kmcontext;