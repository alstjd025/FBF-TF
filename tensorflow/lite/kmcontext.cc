#include <iostream>
#include <cstring>
#include <cmath>

#include "tensorflow/lite/kmcontext.h"

#include "tensorflow/lite/schema/schema_generated.h"
using namespace std;

KmContext kmcontext;

void KmContext::channelPartitioning(vector<pair<int, float>>& layer) {
	vector<int> partitioning_plan;
	vector<float> ratios;
	for (const auto& l : layer) {
		partitioning_plan.push_back(l.first);
		ratios.push_back(l.second);
	}

	channelPartitioning(partitioning_plan, ratios);
}

void KmContext::channelPartitioning(string op_name, float ratio) {
	vector<int> partitioning_plan;
	vector<float> ratios;
	for (int execution_plan_index = 0;
			execution_plan_index < execution_plan_->size(); execution_plan_index++) {
		int node_index = (*execution_plan_)[execution_plan_index];
		
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;
		
		if (strcmp(GetOpName(registration), op_name.c_str()) == 0) {
			partitioning_plan.push_back(execution_plan_index);
			ratios.push_back(ratio);
		}
	}
	
	channelPartitioning(partitioning_plan, ratios);
}

void KmContext::channelPartitioning(std::vector<int>& partitioning_plan, std::vector<float>& ratios) {
	partitioning_plan_.assign(partitioning_plan.begin(), partitioning_plan.end());
	ratios_.assign(ratios.begin(), ratios.end());

	for (int partitioning_plan_index = 0;
		 	partitioning_plan_index < partitioning_plan_.size(); partitioning_plan_index++) {
		int node_index = partitioning_plan_[partitioning_plan_index];
		cout << node_index << endl;
		if (!(node_index < nodes_and_registration_->size())) {
			cerr << "[" << node_index << "] layer is not exist." << endl;
			continue;
		}
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;

		if (strcmp(GetOpName(registration), "TfLiteGpuDelegateV2") == 0) {
			// Unimplemented
			cerr << "[" << node_index << "] TfLiteGpuDelegateV2 ... Unimplemented" << endl;
			continue;
		}

		if (strcmp(GetOpName(registration), "CONV_2D") == 0) {
			for (int n = 1; n < node.inputs->size; ++n) { //change weight tensor 
				int tensor_index = node.inputs->data[n];
				TfLiteTensor& tensor = context_->tensors[tensor_index];
				void** data = &tensor.data.data;
				size_t bytes = tensor.bytes;
				int* dims = (int*)tensor.dims;
				
				if (n == 1) {
					int o = *(dims + 1);
					int w = *(dims + 2);
					int h = *(dims + 3);
					int i = *(dims + 4);
					int next_filter = w * h * i * ((int)bytes / (o * w * h * i));
					next_filter = (int)next_filter * ceil(o * (1 - ratios_[partitioning_plan_index]));
					*data += next_filter; 
				}
				if (n == 2) {							//change bias tensor
					int o = *(dims + 1);
					int next_bias = (int)bytes / o;
					next_bias = (int)next_bias * ceil(o * (1 - ratios_[partitioning_plan_index]));
					*data += next_bias;
				}
				*(dims + 1) *= ratios_[partitioning_plan_index]; //change dim of weight & bias
				int bytes_ = 1;
				for(int i=0; i<tensor.dims->size; i++){ //change bytes of tensor
					bytes_ *= tensor.dims->data[i];
				}
				tensor.bytes = bytes_*sizeof(float);
			}
			for (int n = 0; n < node.outputs->size; ++n) { //change output tensor
				int tensor_index = node.outputs->data[n];
				TfLiteTensor& tensor = context_->tensors[tensor_index];
				int* dims = (int*)tensor.dims;
				int partitioning_dims = (int)floor(*(dims + 4) * ratios_[partitioning_plan_index]);
				*(dims + 4) = partitioning_dims;
				int bytes_ = 1;
				for(int i=0; i<tensor.dims->size; i++){ //change bytes of tensor
					bytes_ *= tensor.dims->data[i];
				}
				tensor.bytes = bytes_*sizeof(float);
			}
		}
		//else if (strcmp(GetOpName(registration), "CONCATENATION") == 0) {	
		//}
		else {
			cerr << "[" << node_index << "] layer have to be CONV_2D or TfLiteGpuDelegateV2" << endl;
			continue;
		}
		
	}
}

void KmContext::printOutputTensors() {
	cout << fixed;
    cout.precision(4);
	for (int partitioning_plan_index = 0;
			partitioning_plan_index < partitioning_plan_.size(); partitioning_plan_index++) {
		int node_index = partitioning_plan_[partitioning_plan_index];
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;
		if (strcmp(GetOpName(registration), "CONV_2D") == 0) {
			int tensor_index = node.outputs->data[0];
			TfLiteTensor& tensor = context_->tensors[tensor_index];
			int* dims = (int*)tensor.dims;
			int o = *(dims + 4);
			//int original_dims = o / ratios_[partitioning_plan_index];
			//int partitioning_dims = original_dims * ratios_[partitioning_plan_index];
			int partitioning_dims = o;
			int parameter = tensor.bytes / 4;
			
			vector<vector<float>> out;
			int w = *(dims + 2);
			int h = *(dims + 3);
			
			for (int i = 0; i < partitioning_dims; ++i) {
				vector<float> o(*(dims + 2) * *(dims + 3));
				out.push_back(o);
			}
		
			//////////////////
			//print raw data//
			//////////////////
	
			/*for (int i = 0; i < parameter; ++i) {
				float data = *((float*)tensor.data.data+i);
				if (data == 0)
					cout << data << " ";
				else
					cout << "\e[92m" << data << "\e[97m" << " ";
				if (i % w == w - 1) cout << endl;
				if (i % (w * h) == w * h - 1) cout << endl;
			} cout << endl;*/

			for (int i = 0; i < parameter * ratios_[partitioning_plan_index]; ++i) {
				float data = *((float*)tensor.data.data+i);
				out[i % partitioning_dims][i / partitioning_dims] = data;
			}	

			for (int i = 0; i < o; ++i) {
				cout << i + 1 << endl;
				int num = 0;
				for (const auto& data : out[i]) {
					if (data == 0)
						cout << /*setw(8) <<*/ data << " ";
					else
						cout << "\e[92m" << /*setw(8) <<*/ data << "\e[97m" << " ";
					if (num % w == w - 1) cout << endl;
					if (num % (w * h) == w * h - 1) cout << endl;
					num += 1;
				}
			} cout << endl;
		}
	}
	cout.setf(ios::fixed);
}

void KmContext::printNodeIndex() {
  for (int execution_plan_index = 0;
		 execution_plan_index < execution_plan_->size(); execution_plan_index++) {
		int node_index = (*execution_plan_)[execution_plan_index];
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;
		cout << endl << GetOpName(registration) << endl;

		cout << "input_index  : ";
		for (int i = 0; i < node.inputs->size; ++i) {
			cout << node.inputs->data[i] << " ";
		} cout << endl;
		
		cout << "output_index : ";
		for (int i = 0; i < node.outputs->size; ++i) {
			cout << node.outputs->data[i] << " ";
		} cout << endl;
	}
}

void KmContext::printNodeDims() {
  string input_shape_[3] = { "input_shape  : ",
								             "filter_shape : ",
								             "bias_shape   : " };
	
	string output_shape_[1] = {"output_shape : " };

  for (int execution_plan_index = 0;
		 execution_plan_index < kmcontext.execution_plan_->size(); execution_plan_index++) {
		int node_index = (*execution_plan_)[execution_plan_index];
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;
		cout << endl << GetOpName(registration) << endl;

		for (int i = 0; i < node.inputs->size; ++i) {
			int tensor_index = node.inputs->data[i];
			int* dims = (int*)context_->tensors[tensor_index].dims;
			cout << input_shape_[i] << "[" << tensor_index << "] ->\t";
			for (int j = 1; j <= *dims; ++j) {
				cout << *(dims + j) << "  ";
			} cout << endl;
		}

		for (int i = 0; i < node.outputs->size; ++i) {
			int tensor_index = node.outputs->data[i];
			int* dims = (int*)context_->tensors[tensor_index].dims;
			cout << output_shape_[i] << "[" << tensor_index << "] ->\t";
			for (int j = 1; j <= *dims; ++j) {
				cout << *(dims + j) << "  ";
			} cout << endl;
		}
	}
}

void KmContext::setContext(TfLiteContext* context, std::vector<int>* execution_plan, 
                    std::vector<std::pair<TfLiteNode, TfLiteRegistration>>* nodes_and_registration) {
    context_ = context;
    execution_plan_ = execution_plan;
    nodes_and_registration_ = nodes_and_registration;
}


 const char* GetOpName(const TfLiteRegistration& op_reg) {
    return tflite::EnumNamesBuiltinOperator()[op_reg.builtin_code];
  }