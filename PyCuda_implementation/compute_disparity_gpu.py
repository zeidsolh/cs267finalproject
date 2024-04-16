import cupy as cp
import numpy as np

def compute_disparity_gpu(L_cpu, R_cpu, block_size_cpu=[9, 9]):
    L_cpu = np.int32(np.array(L_cpu))
    R_cpu = np.int32(np.array(R_cpu))
    [height_cpu, width_cpu] = np.int32(L_cpu.shape)
    block_size_cpu = np.int32(np.array(block_size_cpu))
    D_map_cpu = np.int32(np.zeros(L_cpu.shape))

    # Copying the images to device memory
    L_gpu = cp.asarray(L_cpu)
    R_gpu = cp.asarray(R_cpu)
    block_size_gpu = cp.asarray(block_size_cpu)
    D_map_gpu = cp.asarray(D_map_cpu)

    kernel_code = """
    extern "C" __global__
    void compute_disparity(int *L, int *R, int *block_size, int *D, int L_width, int L_height)
    {	
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int col = blockIdx.x * block_size[0] + tx;
        int row = blockIdx.y * block_size[1] + ty; 

        if((tx >= 0) && (tx < block_size[1]) && (ty >= 0) && (ty < block_size[0]))
        {
            int max_cost = 999999;
            int disparity = 0;

            for(int k = 0; k < 50; k++)
            {
                int cost1 = 0;
                int cost2 = 0;
                int cost3 = 0;

                for(int m = 0; m < block_size[0]; m++)
                {
                    for(int n = 0; n < block_size[1]; n++)
                    {
                        if(((row + m)* L_width + col + n) < L_width * L_height && ((row + m)* L_width + col + n - 25 + k) >= 0 && ((row + m)* L_width + col + n - 25 + k) < L_width * L_height)
                        {
                            cost2 += (L[(row + m)* L_width + col + n] - R[(row + m)* L_width + col + n - 25 + k]) * (L[(row + m)* L_width + col + n] - R[(row + m)* L_width + col + n - 25 + k]) ;
                            cost1 += (L[(row + m)* L_width + col + n] - R[(row + m)* L_width + col + n - 25 + k - 1]) * (L[(row + m)* L_width + col + n] - R[(row + m)* L_width + col + n - 25 + k - 1]) ;
                            cost3 += (L[(row + m)* L_width + col + n] - R[(row + m)* L_width + col + n - 25 + k + 1]) * (L[(row + m)* L_width + col + n] - R[(row + m)* L_width + col + n - 25 + k + 1]) ;
                        }
                    }
                }
                if(cost2 < max_cost)
                {
                    max_cost = cost2;
                    disparity = 25 - k - 0.5 * (cost3 - cost1)/(cost3 + cost1 - cost2) ;
                }
            }
            for(int m = 0; m < block_size[0]; m++)
            {
                for(int n = 0; n < block_size[1]; n++)
                {
                    if(((row + m)* L_width + col + n) < L_width * L_height)
                    {
                        D[(row + m)* L_width + col + n] = disparity;
                    }
                }
            }		
        }
    }
    """

    kernel_name = "compute_disparity"
    kernel_function = cp.RawKernel(kernel_code, kernel_name)

    m = (width_cpu - 1) // 8 + 1
    n = (height_cpu - 1) // 8 + 1
    kernel_function((n, m, 1), (8, 8, 1), (L_gpu, R_gpu, block_size_gpu, D_map_gpu, width_cpu, height_cpu))

    D_map_cpu = D_map_gpu.get()
    D_map_cpu = np.uint8(D_map_cpu)
    D_map_cpu = np.uint8(D_map_cpu * 8)

    return D_map_cpu
