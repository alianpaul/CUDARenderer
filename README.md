# CUDA Renderer
This is a toy circle renderer based on CUDA. I programmed the rendering algorithm of this project. The circle data generation processes are copied from this [CMU project](https://github.com/cmu15418/assignment2/tree/master/render)<br>
I use this project to sharp my CUDA programming knowledge and skill.<br>

The CUDA circle rendering algorithm draw color into the pixel buffer residing in the global memory of the GPU. Then I use OpenGL api to draw the pixel buffer which moved back to CPU.<br> 

## Results
<img src="https://github.com/alianpaul/CUDARenderer/blob/master/Results/100k.png" width="50%" height="50%"><img src="https://github.com/alianpaul/CUDARenderer/blob/master/Results/10k.png" width="50%" height="50%">
<img src="https://github.com/alianpaul/CUDARenderer/blob/master/Results/pattern.png" width="50%" height="50%"><img src="https://github.com/alianpaul/CUDARenderer/blob/master/Results/firework.png" width="50%" height="50%">
## GPU Information
1 CUDA deivce<br>
GeForce GTX 960<br>
* SMs:                    8
* Warp size:              32
* Global mem:             4GB
* Shared mem per block:   48KB
* Shared mem per SM:      96KB
* Max threads per block:  1024
* Max threads per SM:     1024
* Registers per block:    65536
* Registers per SM:       65536

## Optimization Approach 
The main optimization approach I use is Binning. I devide the canvas into N*N bins. N is adjusted by the num of circles. Each bin uses bin-cir pairs to record the circles in this bin.<br>
Then I alleviate the parallelism across the pixels. Each thread block uses shared memory to cache the circle information of the bin it belongs to. Each thread in the block draws its pixel.<br>

## Speedup
I compare the results(rendering time) with the CPU implementation version.The CPU version uses a single thread to draw all the circles one by one.<br>

|Scene  | CPU(ms) |GPU(ms)  | Speedup |
|-------|---------|---------|---------|
|10K    |4550     |280      |16.25    |
|100K   |45353    |2406     |18.84    |
|1217   |90       |21       |4.28     |
|315    |4        |14       |0.28     |


