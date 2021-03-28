#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>

//don't use//#define SFLAG() kmdebug.tokNum(getcwd(kmdir, 256), 3, __FILE__, __func__)

#define SFLAG() kmdebug.tokNum(4, __FILE__, __func__)
#define EFLAG() kmdebug.pop(__FILE__, __func__)
//#define SFLAG() kmdebug.pass()
//#define EFLAG() kmdebug.pass()

// extern original :: interpreter.cc

/*
in future...

depth == 0 -> ERROR!!

i want cout 3 < depth < 5

i want simple file path ex) tensorflow/lite/core/subgraph.cc >> lite/core/subgraph.cc

*/

struct FuncInformation {
	const char* fileName;
	const char* funcName;	
};

class KmDebug {
public:
	KmDebug() {
		FuncInformation f = {
			nullptr,
			nullptr
		};
		info.push_back(f);
	}
    std::vector<FuncInformation> info;
    int depth;
	char* tokNum(int num, const char* filename, const char* funcname) {
		//strtok_r();
		FuncInformation f = {
			filename,
			funcname
		};

		info.push_back(f);

		depth = info.size()-1;
		std::cout << depth << " : ";
 		std::cout << info[depth].fileName << "/"; 
        std::cout << info[depth].funcName << "()\n";	
	};
    void pop(const char *file, const char *func) { print(); info.pop_back(); depth = info.size()-1;};
	void print() { 
		std::cout <<"END " << depth << " : "<< info[depth].fileName << "/";
        std::cout << info[depth].funcName << "()" << std::endl;
	};

	void pass() { return; };
};
extern KmDebug kmdebug;
