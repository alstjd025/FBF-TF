#include <iostream>
#include <vector>
#include <string>

#define DEBUG
#define SFLAG() FunctionFlow func_flow(__FILE__, __PRETTY_FUNCTION__);


struct FuncInformation {
	std::string fileName;
	std::string className;
	std::string funcName;	
};

class FunctionFlow {
  public:
	FunctionFlow(const char* filename, const char* funcname);
	~FunctionFlow();
  private:
	float timer;
};

/*
in future...

depth == 0 -> ERROR!!

i want cout 3 < depth < 5

i want simple file path ex) tensorflow/lite/core/subgraph.cc >> lite/core/subgraph.cc

*/
