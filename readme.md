### Fourier Transform Timers

Fast Fourier Transform and Digital Fourier Transform implementation for Central Processor Unit (CPU) and General Purpose Graphical Processor Unit (NVidia CUDA GPU). Comparing time of execution at the end of program.

#### Implementations:

1. DFT using GPU
2. FFT using GPGPU (any qty of signals that matches powers of 2)
3. DFT using CPU
4. FFT using CPU

#### Comments:

##### 1. FFT using GPGPU

Algorithm can use big length of a signal (more then MaxThreadsPerBlock * 2 items).
There are significant minuses:
This slows down the time of execution.
Why?
- You have to call kernel Log2N + 1 times to synchronize all threads in all blocks (sync device).
- +1 is because you have to call an extra kernel to count W coefficients and reverse bits.
- DFT using GPGPU works faster than FFT implemented that way. This means that GPGPU spends A LOT of time to run a kernel.

So, to get a realy fast FFT you have to put whole signal into one kernel into one block and sync threads.
MaxSignalLength = MaxThreadsPerBlock * 2;
If you have long signal and can't fit it in to one block - ask DSP pro. I am not the one, but i feel that sampling frequency can get lower to fit condition described and it won't effect results of your Fourier Transform.
The Algoritm with one block can be found on the internet i guess.

##### 2. Code

Sorry for not so clean code. I know it can be done better and cleaner.
I was thinking that this way it also may help newcommers to understand it better.

Complex Numbers: 
I was using simple array. No structures. So every Real part sits in even-indexed item and it's Imaginary part sits in next odd-indexed item.

##### P.S. 
i know there is a lot to add and improve - so welcome and thank you very much.

