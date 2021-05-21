#include <unistd.h>
#include <cstring>
#include <ctime>

#include "tensorflow/lite/kmdebug.h"
#include "tensorflow/lite/kmcontext.h"

using namespace std;

std::vector<FuncInformation> info;
int depth;
float execution_time;

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
 	std::cout << info[depth].fileName.substr(16) << "/"; 
	std::cout << info[depth].className << "::";
    std::cout << info[depth].funcName << "()\n";
	timer = (float)clock();
}

FunctionFlow::~FunctionFlow() {
	execution_time += ((float)clock() - timer)/CLOCKS_PER_SEC;
	std::cout <<"END " << depth+1 << " : ";
	std::cout << info[depth].fileName.substr(16) << "/";
	std::cout << info[depth].className << "::";
    std::cout << info[depth].funcName << "() >> " << execution_time << endl;
    info.pop_back();
    depth = info.size()-1;
}