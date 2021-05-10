#include <iostream>
#include <vector>

#define DEBUG
#define SFLAG() FunctionFlow func_flow(__FILE__, __func__);


struct FuncInformation {
	const char* fileName;
	const char* funcName;	
};

class FunctionFlow {
  public:
	FunctionFlow(const char* filename, const char* funcname);
	~FunctionFlow();
};

/*
in future...

depth == 0 -> ERROR!!

i want cout 3 < depth < 5

i want simple file path ex) tensorflow/lite/core/subgraph.cc >> lite/core/subgraph.cc

*/