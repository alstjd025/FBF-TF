#include <unistd.h>
#include <cstring>

#include "tensorflow/lite/kmdebug.h"
#include "tensorflow/lite/kmcontext.h"

using namespace std;

std::vector<FuncInformation> info;
int depth;

FunctionFlow::FunctionFlow(const char* filename, const char* funcname) {
	char temp[500];
	strcpy(temp, funcname);
	vector<char*> tok;
	tok.push_back(strtok(temp, "("));
	tok.push_back(strtok(temp, " "));
	char *funcName, *className;
	while(true) {
		tok.push_back(strtok(NULL, "::"));
		if (tok.back() == NULL) {
			tok.pop_back();
			break;
		}
		className = tok[tok.size()-2];
		funcName = tok.back();
	}

	FuncInformation f = {
		string(filename),
		string(className),
		string(funcName)
	};
	info.push_back(f);

	depth = info.size()-1;
	std::cout << depth+1 << " : ";
 	std::cout << info[depth].fileName << "/"; 
	std::cout << info[depth].className << "::";
    std::cout << info[depth].funcName << "()\n";
}

FunctionFlow::~FunctionFlow() {
	std::cout <<"END " << depth+1 << " : ";
	std::cout << info[depth].fileName << "/";
	std::cout << info[depth].className << "::";
    std::cout << info[depth].funcName << "()" << endl;
    info.pop_back();
    depth = info.size()-1;
}