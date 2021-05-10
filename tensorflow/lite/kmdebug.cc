#include <unistd.h>
#include <string.h>

#include "tensorflow/lite/kmdebug.h"
#include "tensorflow/lite/kmcontext.h"

using namespace std;

std::vector<FuncInformation> info;
int depth;

FunctionFlow::FunctionFlow(const char* filename, const char* funcname) {
	FuncInformation f = {
		filename,
		funcname
	};
	info.push_back(f);

	depth = info.size()-1;
	cout << depth+1 << " : ";
 	cout << info[depth].fileName << "/"; 
    cout << info[depth].funcName << "()\n";
}

FunctionFlow::~FunctionFlow() {
	cout <<"END " << depth+1 << " : "<< info[depth].fileName << "/";
    cout << info[depth].funcName << "()" << endl;
    info.pop_back();
    depth = info.size()-1;
}