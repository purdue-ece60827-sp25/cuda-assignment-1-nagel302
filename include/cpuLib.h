
/**
 * @file cpuLib.h
 * @author Jake Nagel (nagel30@purdue.edu)
 * @brief 
 * @version 0.2
 * @date 2025-01-29
 * 
 * 0.2 : Added tolerance to verifyVector
 *
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef CPU_LIB_H
#define CPU_LIB_H

	#include <iostream>
	#include <cstdlib>
	#include <ctime>
	#include <random>
    #include <iomanip>
	#include <chrono>
	#include <cstring>
	#include <cstdarg>
	
	#define SAXPY_TOL 0.001

	// Uncomment this to suppress console output
	#define DEBUG_PRINT_DISABLE

	extern void dbprintf(const char* fmt...);


	extern void vectorInit(float* v, int size);
	extern int verifyVector(float* a, float* b, float* c, float scale, int size);
	extern void printVector(float* v, int size);
	
	extern void saxpy_cpu(float* x, float* y, float scale, uint64_t size);

	extern int runCpuSaxpy(uint64_t vectorSize);

	extern int runCpuMCPi(uint64_t iterationCount, uint64_t sampleSize);

#endif
