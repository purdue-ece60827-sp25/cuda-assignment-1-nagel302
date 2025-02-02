/**
 * @file cudaLib.cu
 * @author Jake Nagel (nagel30@purdue.edu)
 * @brief 
 * @version 0.1
 * @date 2025-01-29
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;	
	if (i < size) {
		y[i] = scale * x[i] + y[i];
	}
}


int runGpuSaxpy(int vectorSize) {
	size_t size = vectorSize * sizeof(float);
	int threads_per_block = 256;
    
	// Allocate Host Memory for float vectors x and y
	float *x_h = (float*) malloc(size);
	float *y_h = (float*) malloc(size);
	float *y_original = (float*) malloc(size);  // For saving original value of y

	// Generate a random scaling factor
	std::random_device rand_seed;                                // Seed Generator
	std::mt19937 gen(rand_seed());                               // Mersenne Twister
	std::uniform_real_distribution<float> floatdist(0.0, 100.0); // Define the range (arbitrarily [0.0:99.0])
    float scale = floatdist(gen);                                // Get the random scale

	// Populate vectors with random data using provided CPU Function.
	vectorInit(x_h, vectorSize);
	vectorInit(y_h, vectorSize);

	// Save original y vector for later verification. SAXPY overwrites 
	// the Y we provide it.	
	for (int idx = 0; idx < vectorSize; idx++)
	{
		y_original[idx] = y_h[idx];
	}

	// Allocate Memory on the GPU for vectors x and y
    float *x_d;
	float *y_d;
	cudaMalloc((void**) &x_d, size);
	cudaMalloc((void**) &y_d, size);

	// Copy the vectors x and y to the device
	cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice);

	//	Run the SAXPY GPU
	saxpy_gpu<<<ceil((vectorSize + (threads_per_block-1))/threads_per_block),threads_per_block>>>(x_d, y_d, scale, vectorSize);

	// Copy the resulting 'y' vector from the GPU to host
	cudaMemcpy(y_h, y_d, size, cudaMemcpyDeviceToHost); 

	// Free the Device Memory used
	cudaFree(x_d);
	cudaFree(y_d);

	// Verify the Vector Results
	int errors = verifyVector(x_h, y_original, y_h, scale, vectorSize);
	std::cout << "There were " << errors << " errors found.";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for 
 sampleSize points. 

 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The 
 length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < pSumSize) {		
		float x, y;
		curandState_t rng;
		curand_init(clock64(), i, 0, &rng);

		for (uint64_t k=0; k<sampleSize; k++) {
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);
			if (int(x*x + y*y) == 0) {
				pSums[i]++;
			}
		}
	}
}

// There is no supported atomicAdd that works with uint64_t type, so making a new one here
__device__ uint64_t atomicAddUint64(uint64_t* address, uint64_t val) {
    uint64_t old = *address, assumed;

    do {
        assumed = old;
        old = atomicCAS((unsigned long long int*)address, assumed, assumed + val);
    } while (assumed != old);

    return old;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	extern __shared__ unsigned long shared_data[];

	// each thread adds a chunk of the sums into an element in shared mem
	uint64_t tid = threadIdx.x;
	uint64_t i = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Initialize Shared Mem
	shared_data[tid] = 0;
	__syncthreads();

    // Load partial sum element to shared memory
	if (i < pSumSize) {
		atomicAddUint64(&shared_data[tid % reduceSize], pSums[i]);
	}
	__syncthreads();

	// Perform the reduced addition
	for (uint16_t k = blockDim.x / 2; k > 0; k >>= 1) {
		if (tid < k) {
			shared_data[tid] += shared_data[tid + k];
		}
		__syncthreads();
	}

	// Finally write the result from the reduced sum to global memory (in totals)
	if (tid < reduceSize && tid % blockDim.x == 0) {
		atomicAddUint64(&totals[tid], shared_data[tid]);
	}

}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds." << "\n";

	return 0;
}


double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	uint64_t pSumsSize = generateThreadCount * sizeof(uint64_t);
	uint64_t totalHits = 0;
	double approxPi = 0;

	// Allocate host memory to copy array of sums
	uint64_t *pSums_h = (uint64_t*) malloc(pSumsSize);

	// Allocate device memory for populating array of sums
	uint64_t *pSums_d;	
	cudaMalloc((void**) &pSums_d, pSumsSize);

	// Run Monte Carlo PI GPU - 256 threads per block fixed
	generatePoints<<<ceil((generateThreadCount + 255)/256), 256>>>(pSums_d, generateThreadCount, sampleSize);

    // Copy array of sums to host and free device memory
	cudaMemcpy(pSums_h, pSums_d, pSumsSize, cudaMemcpyDeviceToHost);
	cudaFree(pSums_d);

	// Sum up all the total hits
	for (uint64_t k=0; k<generateThreadCount; k++) {
		totalHits += pSums_h[k];
	}

	// Compute Pi from the result
	approxPi = ((double) totalHits / sampleSize ) / generateThreadCount;
	approxPi = approxPi * 4.0f;

	return approxPi;
}


/*
// Attempt to use the reduceCount
double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	uint64_t pSumsSize = generateThreadCount * sizeof(uint64_t);
	uint64_t tSumsSize = reduceSize * sizeof(uint64_t);
	uint64_t totalHits = 0;
	int threads_per_block = 256;
	double approxPi = 0;

	// Allocate host memory to copy array of sums
	uint64_t *tSums_h = (uint64_t*) malloc(tSumsSize);

	// Allocate device memory for populating arrays of sums
	uint64_t *pSums_d, *tSums_d;	
	cudaMalloc((void**) &pSums_d, pSumsSize);
	cudaMalloc((void**) &tSums_d, tSumsSize);

	// Run Monte Carlo PI GPU
	generatePoints<<<ceil((generateThreadCount + (threads_per_block-1))/threads_per_block), threads_per_block>>>(pSums_d, generateThreadCount, sampleSize);    
	cudaDeviceSynchronize();

    // Run the Reduced Sums GPU
	reduceCounts<<<ceil((reduceThreadCount + (threads_per_block-1))/threads_per_block), threads_per_block, (reduceSize * sizeof(uint64_t))>>>(pSums_d, tSums_d, generateThreadCount, reduceSize);

    // Copy array of sums to host and free device memory
	cudaMemcpy(tSums_h, tSums_d, tSumsSize, cudaMemcpyDeviceToHost);
	cudaFree(pSums_d);
	cudaFree(tSums_d);

	// Sum up the total hits in a reduced array size - this is a much smaller loop for host
	for (uint64_t k=0; k<reduceSize; k++) {
		totalHits += tSums_h[k];
	}

	// Compute Pi from the result
	approxPi = 4.0f * (double) totalHits / generateThreadCount;
	approxPi = approxPi * 4.0f;

	return approxPi;
}
*/