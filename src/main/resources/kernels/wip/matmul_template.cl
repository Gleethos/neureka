

kernel void multiply(global float* pC, global const float* pA, global const float* pB, int M, int N, int P)
{
    local float shA[16][16];
    local float shB[16][16];
    int m = get_global_id(0);
    int p = get_global_id(1);
    int pc = (get_group_id(1)<<4)+get_local_id(0);
    float result = 0.0;
    for (int n = get_local_id(1); n < N; n += 16)
    {
        shA[get_local_id(0)][get_local_id(1)] = pA[(N*m)+n];
        shB[get_local_id(0)][get_local_id(1)] = pB[(P*n)+pc];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < 16; i++)
        {
            result += (shA[get_local_id(0)][i]*shB[get_local_id(1)][i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    pC[(P*m)+p] = result;
}