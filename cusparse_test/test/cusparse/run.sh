nvcc timing_sddmm.cu -lcublas -lcusparse -o run_it 2>build.err
CUDA_VISIBLE_DEVICES=3 ./run_it