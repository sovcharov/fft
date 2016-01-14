#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
// #include <cufft.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cstdio>
using namespace std;

__global__ void countW(float * W)
{
	float pi = 3.1415926535897932384626433832795;
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int N=blockDim.x*gridDim.x*2;//*2 couse we call function with half of N
	W[i*2]=cos(2*pi*i/N);
	W[i*2+1]=-sin(2*pi*i/N);
}

__global__ void myCudaFFTBitReverse(const float *signal, float * output, float * W)
{
	unsigned int v = blockIdx.x*blockDim.x+threadIdx.x; 
	unsigned int k = v;     // reverse the bits in this
	unsigned int t = 0;     // t will have the reversed bits of v
	float N=blockDim.x*gridDim.x;
	N=log2f(N);
	for (int i = N; i; i--)
	{
	  t <<= 1;
	  t |= v & 1;
	  v >>= 1;
	}
	output[k*2]=signal[t*2];
	output[k*2+1]=signal[t*2+1];
}

__global__ void myCudaFFTBitReverseAndWCount(const float *signal, float * output, float * W)
{
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x; 
	unsigned int v = i;       // reverse the bits in this
	unsigned int t = 0;     // t will have the reversed bits of v
	float N=blockDim.x*gridDim.x;
	float n=log2f(N);
	for (int k = n; k; k--)
	{
	  t <<= 1;
	  t |= v & 1;
	  v >>= 1;
	}
	output[i*2]=signal[t*2];
	output[i*2+1]=signal[t*2+1];
	if(i<N/2){
		float pi = 3.1415926535897932384626433832795;
		W[i*2]=cos(2*pi*i/(N));
		W[i*2+1]=-sin(2*pi*i/(N));
	}
}

__global__ void myCudaDFT(const float *signal, float * output)
{
	float pi = 3.1415926535897932384626433832795;
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int N=blockDim.x*gridDim.x;
	output[i*2] = 0;
	output[i*2+1] = 0;
	for(int j=0;j<N;j++){
		output[i*2]+=signal[j*2]*cos(2*pi*i*j/N);
		output[i*2+1]+=-signal[j*2]*sin(2*pi*i*j/N);
	}
}

void myFFT(float * output, const int NTotal, const int NCurrent, const float *W){
	if(NCurrent>1){
		float * tempOutput = new float[NTotal*2];
		//this part was writen for test
		for(int i=0;i<NTotal/2;i++){
				//cout<<endl<<(int)truncf(i/(NCurrent/2))<<"----"<<i%(NCurrent/2);
				int indexTempOutputEven=((int)i/(NCurrent/2)*NCurrent/2*2+i%(NCurrent/2))*2;
				int indexTempOutputOdd=indexTempOutputEven+NCurrent/2*2;
				int indexOutputEven = (i)*2*2;
				int indexOutputOdd = indexOutputEven+1*2;
				//cout<<endl<<indexTempOutputEven<<"--"<<indexTempOutputOdd<<"______"<<indexOutputEven<<"--"<<indexOutputOdd;
				tempOutput[indexTempOutputEven]=output[indexOutputEven];
				tempOutput[indexTempOutputEven+1]=output[indexOutputEven+1];//img part
				tempOutput[indexTempOutputOdd]=output[indexOutputOdd];
				tempOutput[indexTempOutputOdd+1]=output[indexOutputOdd+1];//img part
		}
		// for(int i=0;i<NTotal/NCurrent;i++){
		// 	for(int k=0;k<NCurrent/2;k++){
		// 		int indexTempOutputEven=(i*NCurrent/2*2+k)*2;
		// 		int indexTempOutputOdd=indexTempOutputEven+NCurrent/2*2;
		// 		int indexOutputEven = (i*NCurrent/2+k)*2*2;
		// 		int indexOutputOdd = indexOutputEven+1*2;
		// 		// cout<<endl<<indexTempOutputEven<<"--"<<indexTempOutputOdd<<"______"<<indexOutputEven<<"--"<<indexOutputOdd;
		// 		tempOutput[indexTempOutputEven]=output[indexOutputEven];
		// 		tempOutput[indexTempOutputEven+1]=output[indexOutputEven+1];//img part
		// 		tempOutput[indexTempOutputOdd]=output[indexOutputOdd];
		// 		tempOutput[indexTempOutputOdd+1]=output[indexOutputOdd+1];//img part
		// 	}
		// }		
		for(int i=0;i<NTotal;i++){
			output[i*2]=tempOutput[i*2];
			output[i*2+1]=tempOutput[i*2+1];
		}
		// cout<<endl;
		// cout<<"New Output:"<<endl;
		// for(int i=0;i<NTotal;i++){
		// 	cout<<output[i*2]<<" + "<<output[i*2+1]<<", ";
		// }
		myFFT(output,NTotal,NCurrent/2,W);
		// cout<<endl<<"NOW COMPUTATION:"<<endl;
		// cout<<endl;
		// cout<<"New Input:"<<endl;
		// for(int i=0;i<NTotal;i++){
		// 	cout<<output[i*2]<<" + "<<output[i*2+1]<<", ";
		// }		
		for(int i=0;i<NTotal/NCurrent;i++){
			for(int k=0;k<NCurrent/2;k++){
				///////////////////////////////////////////////////////////////
				//here is thee part of complex numbers addition and multiplication
				///////////////////////////////////////////////////////////////
				int indexTempOutputEven=(i*NCurrent/2*2+k)*2;
				int indexTempOutputOdd=indexTempOutputEven+NCurrent/2*2;
				int indexW = k*2*NTotal/NCurrent;
				// cout<<endl<<indexTempOutputEven<<"--"<<indexTempOutputOdd<<"______"<<indexOutputEven<<"--"<<indexOutputOdd<<" indexW: "<<indexW<<" NCurrent: "<<NCurrent<<endl;
				tempOutput[indexTempOutputEven]=output[indexTempOutputEven]+output[indexTempOutputOdd]*W[indexW]-output[indexTempOutputOdd+1]*W[indexW+1];
				tempOutput[indexTempOutputEven+1]=output[indexTempOutputEven+1]+output[indexTempOutputOdd]*W[indexW+1]+output[indexTempOutputOdd+1]*W[indexW];//img part
				tempOutput[indexTempOutputOdd]=output[indexTempOutputEven]-(output[indexTempOutputOdd]*W[indexW]-output[indexTempOutputOdd+1]*W[indexW+1]);
				tempOutput[indexTempOutputOdd+1]=output[indexTempOutputEven+1]-(output[indexTempOutputOdd]*W[indexW+1]+output[indexTempOutputOdd+1]*W[indexW]);//img part
				output[indexTempOutputEven]=tempOutput[indexTempOutputEven];
				output[indexTempOutputEven+1]=tempOutput[indexTempOutputEven+1];
				output[indexTempOutputOdd]=tempOutput[indexTempOutputOdd];
				output[indexTempOutputOdd+1]=tempOutput[indexTempOutputOdd+1];
			}
		}
	}
}

void setOutputAsSignal(const float * signal, float * output, int N){
	for(int i=0;i<N;i++){
		output[i*2]=signal[i*2];
		output[i*2+1]=signal[i*2+1];
	}	
}

float niceNumbers(float x){
	if(abs(x)<0.001) x = 0;
	return x;
}

__global__ void myCudaFFT(const float *signal, float * output, float * W, int NCurrent)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int N = blockDim.x*gridDim.x;
	int indexW = i%NCurrent*N/NCurrent*2;
	int upIndex = ((int)i/NCurrent*NCurrent*2+i%NCurrent)*2;
	int downIndex = upIndex+NCurrent*2;
	output[upIndex]=signal[upIndex]+signal[downIndex]*W[indexW]-signal[downIndex+1]*W[indexW+1];//real of upper 	one
	output[upIndex+1]=signal[upIndex+1]+signal[downIndex]*W[indexW+1]+signal[downIndex+1]*W[indexW];//img of upper one
	output[downIndex]=signal[upIndex]-(signal[downIndex]*W[indexW]-signal[downIndex+1]*W[indexW+1]);//real of lower one
	output[downIndex+1]=signal[upIndex+1]-(signal[downIndex]*W[indexW+1]+signal[downIndex+1]*W[indexW]);//img of lower one	
}

int main()
{
	float pi = 3.1415926535897932384626433832795;
	//setting max threads for device
	cudaDeviceProp devProp;
	cudaGetDeviceProperties (&devProp, 0);
	int threads = devProp.maxThreadsPerBlock;
	threads = 2;	
	int blocks = 2;
	int N = threads*blocks;
	//N=4;//set manually N for test
	//set timers variables for cpu ang gpu
	int timeDFTCPU = 0, timeFFTCPU = 0, start_time =0;
	float timeDFTGPGPU=0.0f, timeFFTGPGPU = 0.0f ;
	
	//setting arrays of signals and output(matrix 2^n x 2)
	float *signal = new float [N*2];// "*2" means it has imaginary part
	float *output = new float [N*2];
	float *W= new float[N];	//declare W for FFT
	for(int i=0;i<N;i++){
		signal[i*2]=sin((float)i/10);
		signal[i*2+1]=0.0f;
		output[i*2]=0.0f;
		output[i*2+1]=0.0f;
	}
	
	//set test data start
	// signal[0]=1;
	// signal[2]=2;
	// signal[4]=3;
	// signal[6]=1;
	//set test data end

	// test print of signal
	// for(int i=0;i<N;i++){
	// 	cout<<signal[i*2]<<','<<signal[i*2+1]<<endl;
	// }
	// return 0;	
	// end of test

	////////////////////////////////////////////////////////////////
	// CUDA PART
	////////////////////////////////////////////////////////////////
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	float *dev_W = 0;
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_a, N*2 * sizeof(float));
	cudaMemcpy(dev_a, signal, N*2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_b, N*2 * sizeof(float));
	cudaMalloc((void**)&dev_c, N*2 * sizeof(float));
	cudaMalloc((void**)&dev_W, N*2/2 * sizeof(float));
	//DFT GPGPU start
	// синхронизация нитей GPU через event
	cudaEvent_t start, stop;
	cudaEventCreate(&start); //Создаем event
	cudaEventCreate(&stop); //Создаем event
	cudaEventRecord(start, 0); //Записываем event
	
	myCudaDFT<<<blocks, threads>>>(dev_a, dev_b);

	cudaEventRecord(stop, 0); //Записываем event
	cudaEventSynchronize(stop); //Синхронизируем event
	cudaEventElapsedTime(&timeDFTGPGPU,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//DFT GPGPU end
	cudaMemcpy(output, dev_b, N*2 * sizeof(float), cudaMemcpyDeviceToHost);

	//test print of output
	// cout<<endl<<"CUDA DFT:"<<endl;
	// for(int i=0;i<N;i++){
	// 	cout<<niceNumbers(output[i*2])<<" + "<<niceNumbers(output[i*2+1])<<"i"<<endl;
	// }
	//end of test	

	//FFT GPGPU start
	cudaEventCreate(&start); //Создаем event
	cudaEventCreate(&stop); //Создаем event
	cudaEventRecord(start, 0); //Записываем event
	
	myCudaFFTBitReverseAndWCount<<<blocks, threads>>>(dev_a, dev_b, dev_W);
	for(int NCurrent=1;NCurrent<=N/2;NCurrent*=2){
		myCudaFFT<<<blocks/2, threads>>>(dev_b, dev_c, dev_W,NCurrent);	
		if (NCurrent!=N/2) cudaMemcpy(dev_b, dev_c, N*2 * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	cudaEventRecord(stop, 0); //Записываем event
	cudaEventSynchronize(stop); //Синхронизируем event
	cudaEventElapsedTime(&timeFFTGPGPU,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//FFT GPGPU end

	cudaMemcpy(output, dev_c, N*2 * sizeof(float), cudaMemcpyDeviceToHost);

	//test print of output
	cout<<endl<<"CUDA FFT:"<<endl;
	for(int i=0;i<N;i++){
		cout<<niceNumbers(output[i*2])<<" + "<<niceNumbers(output[i*2+1])<<"i"<<endl;
	}
	//end of test	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_W);

	// cufftHandle plan;
	// cufftComplex *data;
	// cudaMalloc((void**)&data, sizeof(cufftComplex)*N);
	// cufftPlan1d(&plan, N, CUFFT_C2C, 1);
	// cufftExecC2C(plan, data, data, CUFFT_FORWARD); 
	// cufftDestroy(plan); 
	// cudaFree(data); 

	///////////////////////////////////////////////////////////////
	// END OF CUDA
	///////////////////////////////////////////////////////////////

	//DFT CPU start
	start_time = clock();
	for(int i=0;i<N;i++){
		output[i*2]=0;
		output[i*2+1]=0;
		for(int j=0;j<N;j++){
			output[i*2]+=signal[j*2]*cos(2*pi*i*j/N);
			output[i*2+1]+=-signal[j*2]*sin(2*pi*i*j/N);
		}
	}
	timeDFTCPU = (clock() - start_time) / (float)CLOCKS_PER_SEC * 1000.0;
	//DFT CPU end

	//test print of output
	cout<<endl<<"DFT:"<<endl;
	for(int i=0;i<N;i++){
		cout<<niceNumbers(output[i*2])<<" + "<<niceNumbers(output[i*2+1])<<"i"<<endl;
	}
	//end of test

	//setting output as signals array, it will be changed after FFT
	setOutputAsSignal(signal,output,N);
	//end of setting back signal

	//FFT CPU start (FFT with call of Log2N calls)
	start_time = clock();
	//count all W^0:N/2
	for(int i = 0;i<N/2;i++){
		W[i*2]=cos(2*pi*i/N);
		W[i*2+1]=-sin(2*pi*i/N);
	}
	int * indexes = new int [N];
	for(int i=0;i<N;i++){
		indexes[i]=i;
	}
	myFFT(output,N,N,W);
	timeFFTCPU = (clock() - start_time) / (float)CLOCKS_PER_SEC * 1000.0;
	//FFT CPU end


	//test print of W
	// cout<<endl<<"W array:"<<endl;
	// for(int i=0;i<N/2;i++){
	// 	cout<<niceNumbers(W[i*2])<<" + "<<niceNumbers(W[i*2+1])<<"i"<<endl;
	// }
	// cout<<endl;
	//end of test

	//test print of output
	// cout<<endl<<"FFT:"<<endl;
	// for(int i=0;i<N;i++){
	// 	cout<<niceNumbers(output[i*2])<<" + "<<niceNumbers(output[i*2+1])<<"i"<<endl;
	// }
	//end of test

	//print timers
	cout<<"================================================================"<<endl;
	cout<<"CPU TIMERS: "<<endl;
	cout<<"DFT: "<<timeDFTCPU<<" ms."<<endl;
	cout<<"FFT: "<<timeFFTCPU<<" ms."<<endl;
	cout<<endl;
	cout<<"GPGPU TIMERS: "<<endl;
	cout<<"DFT: "<<timeDFTGPGPU<<" ms."<<endl;
	cout<<"FFT: "<<timeFFTGPGPU<<" ms."<<endl;
	cout<<"================================================================"<<endl;
	getchar();// uncomment for VS
	return 0;
}