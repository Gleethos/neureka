package it.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.CLContext
import testutility.opencl.DispatchUtility
import neureka.dtype.DataType
import neureka.ndim.config.NDConfiguration
import neureka.view.TsrStringSettings
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("The OpenCLDevice Specification")
@Narrative('''

    Tensors need devices for execution!
    By default we use the CPU as a default device, but sometimes we want to
    use something more suitable for large amounts of data an a high degree of parallelization.
    This is were the OpenCLDevice comes into play!
    It is a Device implementation built on top of the JOCL library, a thin OpenCL API!
    We expect the OpenCLDevice to stored tensors while still being able to read and write
    data from and to stored tensors.
    Also, an OpenCLDevice should allows us to compile OpenCL kernel code on the fly...
                    
''')
class OpenCLDevice_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2> OpenCLDevice Integration Tests </h2>
            <p>
                Specified below are strict tests covering the behavior
                of the OpenCLDevice when hosting tensors and executing 
                operations on them.
            </p>
        """
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'An OpenCLDevice loads tensors in a provided lambda temporarily.'()
    {
        given: 'The first found OpenCLDevice instance.'
            Device<?> device = Device.find('first')
        and : 'We create 2 tensors with different default devices.'
            Tsr<?> t = Tsr.of([4, 3], 2)
            Tsr<?> s = Tsr.of([3, 2], -1).to(device)

        expect : 'At first, both tensors are stored where we said they should be.'
            !device.has(t)
            device.has(s)
        and : 'The two tensors should also know to which devices they belong!'
            t.device === CPU.get()
            s.device === device

        and : 'When we check their location in the lambda we expect them both to be on the device!'
            device.borrow(t, s)
                   .in(() -> {
                       return device.has(t) && device.has(s)
                   })

        and : 'After the lambda ran, we expect everything to be reverted.'
            !device.has(t)
            device.has(s)
        and : 'The two tensors should also know to which devices they belong!'
            t.device === CPU.get()
            s.device === device

    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'The "getValue()" method of an outsourced tensor will return the expected array type.'()
    {
        given : 'A new tensor.'
            Tsr t = Tsr.ofDoubles().withShape( 1, 2 ).all(0)

        expect : 'This tensor is initially of type "Double", meaning it is backed by a "double[]" array internally...'
            t.dataType == DataType.of( Double.class )

        when : 'The tensor is being transferred to the first found OpencCLDevice...'
            Device.find('first').store( t )

        then : 'The data type of the tensor is being converted to single precision.'
            t.dataType == DataType.of( Float.class )

        when : 'The tensor value is being fetched...'
            def value = t.getValue()

        then : 'This value object is an instance of a "float[]" array because the device converted the value.'
            value instanceof float[]

        //when : 'The tensor datatype is being change from double to float...'
        //    t.to32()
        //    data = t.getValue()
        //then : 'The data that will be returned by the tensor is of type "float[]".'
        //    value instanceof float[] // WIP!
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'The "getData()" method of an outsourced tensor will return null when outsourced.'()
    {
        given : 'A new tensor belonging to the first found OpenCLDevice instance.'
            Tsr t = Tsr.ofDoubles().withShape( 1, 2 ).all(0)

        expect : 'The tensor start with having data stored within.'
            t.unsafe.data != null

        when : 'The tensor is being stored on the device...'
            t.to(Device.find('first'))
        and : 'The tensor value is being fetched...'
            def data = t.unsafe.data

        then : 'The value variable is null.'
            data == null
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc compilation produces executable kernel.'() {

        given :
            def device = Neureka.get().backend().get(CLContext.class).getPlatforms()[0].devices[0]
            def someData  = Tsr.of( new float[]{ 2, -5, -3, 9, -1 } ).to( device )

        expect : 'The OpenCL device initially does not have the "dummy_kernel" we are going to create.'
            !device.hasAdHocKernel( 'dummy_kernel' )

        when : 'Compiling a kernel called "dummy_kernel"...'
            device.compileAdHocKernel( 'dummy_kernel', """
                __kernel void dummy_kernel (
                        __global float* output,
                        __global float* input,
                        float value 
                    ) { 
                        unsigned int i = get_global_id( 0 );
                        output[i] = input[i] + value; 
                    }
                """
            )

        then : 'The OpenCL device contrary to before now has a kernel by this name.'
            device.hasAdHocKernel( 'dummy_kernel' )

        and : 'The device still references the source code od that kernel.'
            device._kernelCache._adhocKernels['dummy_kernel'].source == """
                __kernel void dummy_kernel (
                        __global float* output,
                        __global float* input,
                        float value 
                    ) { 
                        unsigned int i = get_global_id( 0 );
                        output[i] = input[i] + value; 
                    }
                """

        when : 'Executing the kernel by passing the previously defined tensor...'
            device.getAdHocKernel( 'dummy_kernel' )
                    .pass( someData )
                    .pass( someData )
                    .pass( -4f )
                    .call( someData.size() )

        then : 'Each value within the tensor will all have the number 4 subtracted from it.'
            someData.toString() == "(5):[-2.0, -9.0, -7.0, 5.0, -5.0]"

    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc compilation works for WIP general purpose matrix multiplication.'(
           int regSize, int locSize, int M, int K, int N, String expected
    ) {

        given :
            var device = Neureka.get().backend().get(CLContext.class).getPlatforms()[0].devices[0]
            var kernelName = "dummy_mm_${M}x${K}x${N}"
            var params = DispatchUtility.findBestParams(locSize, regSize, K, M, N)

            //{max_ts_row, max_ts_col, max_ts_com, max_wpt_row, max_wpt_col}
            var TSM  = params[0] //4   //= 128
            var TSN  = params[1] //4   //= 128
            var TSK  = params[2] //2   //= 16
            var WPTM = params[3] // 2  //=  8
            var WPTN = params[4] // 2  //=  8

            long[] local=   new long[]{ TSM/WPTM, TSN/WPTN }
            long[] global = new long[]{ (M/WPTM), (N/WPTN) }
            //println('TSM:'+TSM+' - TSN:'+TSN+' - TSK:'+TSK+' - WPTM'+WPTM+' - WPTN:'+WPTN+' |'+local+'|'+global)

            Tsr A = Tsr.of( [M,K], 0f )
            Tsr B = Tsr.of( [K,N], 0f )
            Tsr C = Tsr.of( [M,N], 0f )
            A[0..M-1,0..K-1] = Tsr.of([M,K], 3..1)
            B[0..K-1,0..N-1] = Tsr.of([K,N], -5..0)

            var reference = A.matMul(B).value // CPU execution for reference!

            A.to( device )
            B.to( device )
            C.to( device ).setIsVirtual(false)

        expect :
            !device.hasAdHocKernel( kernelName )

        when :
            device.compileAdHocKernel( kernelName, """
                    #define TSM ${TSM}                   // The tile-size in dimension M
                    #define TSN ${TSN}                   // The tile-size in dimension N
                    #define TSK ${TSK}                   // The tile-size in dimension K
                    #define WPTM ${WPTM}                 // The work-per-thread in dimension M
                    #define WPTN ${WPTN}                 // The work-per-thread in dimension N
                    #define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M
                    #define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N
                    #define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
                    #define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

                    // Use 2D register blocking (further increase in work per thread)
                    __kernel void $kernelName(
                        const int M, const int N, const int K,
                        const __global float* A,
                        const __global float* B,
                              __global float* C
                    ) {
                        
                        // Thread identifiers
                        const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
                        const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
                        const int offsetM = TSM*get_group_id(0); // Work-group offset
                        const int offsetN = TSN*get_group_id(1); // Work-group offset
                     
                        // Local memory to fit a tile of A and B
                        __local float Asub[TSK][TSM];
                        __local float Bsub[TSN][TSK+2];
                     
                        // Allocate register space
                        float Areg;
                        float Breg[WPTN];
                        float acc[WPTM][WPTN];
                     
                        // Initialise the accumulation registers
                        for ( int wm=0; wm<WPTM; wm++ ) {
                            for ( int wn=0; wn<WPTN; wn++ ) {
                                acc[wm][wn] = 0.0f;
                            }
                        }
                        
                        // Loop over all tiles
                        int numTiles = K/TSK;
                        for ( int t=0; t<numTiles; t++ ) {
                     
                            // Load one tile of A and B into local memory
                            for ( int la=0; la<LPTA; la++ ) {
                                int tid = tidn*RTSM + tidm;
                                int id = la*RTSN*RTSM + tid;
                                int row = id % TSM;
                                int col = id / TSM;
                                int tiledIndex = TSK*t + col;
                                Asub[col][row] = A[tiledIndex*M + offsetM + row];
                                Bsub[row][col] = B[tiledIndex*N + offsetN + row];
                            }
                            
                            // Synchronise to make sure the tile is loaded
                            barrier(CLK_LOCAL_MEM_FENCE);
                     
                            // Loop over the values of a single tile
                            for ( int k=0; k<TSK; k++ ) {
                     
                                // Cache the values of Bsub in registers
                                for ( int wn=0; wn<WPTN; wn++ ) {
                                    int col = tidn + wn*RTSN;
                                    Breg[wn] = Bsub[col][k];
                                }
                     
                                // Perform the computation
                                for ( int wm=0; wm<WPTM; wm++ ) {
                                    int row = tidm + wm*RTSM;
                                    Areg = Asub[k][row];
                                    for ( int wn=0; wn<WPTN; wn++ ) {
                                        acc[wm][wn] += Areg * Breg[wn];
                                    }
                                }
                            }
                     
                            // Synchronise before loading the next tile
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                     
                        // Store the final results in C
                        for ( int wm=0; wm<WPTM; wm++ ) {
                            int globalRow = offsetM + tidm + wm*RTSM;
                            for ( int wn=0; wn<WPTN; wn++ ) {
                                int globalCol = offsetN + tidn + wn*RTSN;
                                //C[globalCol*M + globalRow] = acc[wm][wn];
                                C[globalCol + N*globalRow] = acc[wm][wn];
                            }
                        }
                    }
                    """
            )

        then :
            device.hasAdHocKernel( kernelName )


        when :
            device.getAdHocKernel( kernelName )
                    .pass( M ).pass( N ).pass( K )
                    .pass( A ).pass( B ).pass( C )
                    .call( global, local )

        then :
            C.value == reference // GPU should produce the same as CPU!
        and :
            expected == null || C.toString({it.setRowLimit(50)}) == expected

        where :
             regSize | locSize | M  | K  | N  || expected
               2**2  |  4**2   | 4  | 4  | 4  || '(4x4):[-35.0, -26.0, -29.0, -20.0, -30.0, -22.0, -20.0, -12.0, -19.0, -12.0, -23.0, -16.0, -35.0, -26.0, -29.0, -20.0]'
                16   |  32     | 16 | 16 | 16 || '(16x16):[-115.0, -82.0, -109.0, -76.0, -73.0, -40.0, -115.0, -82.0, -109.0, -76.0, -73.0, -40.0, -115.0, -82.0, -109.0, -76.0, -110.0, -78.0, -76.0, -44.0, -102.0, -70.0, -110.0, -78.0, -76.0, -44.0, -102.0, -70.0, -110.0, -78.0, -76.0, -44.0, -75.0, -44.0, -103.0, -72.0, -101.0, -70.0, -75.0, -44.0, -103.0, -72.0, -101.0, -70.0, -75.0, -44.0, -103.0, -72.0, -115.0, -82.0, ... + 206 more]'
               2**2  |  4**2   | 4  | 16 | 8  || null
               2**2  |  4**2   | 16 | 16 | 32 || null

    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc matrix multiplication works for multiple of 16 matrices.'(
            int seed, int M, int K, int N, String expected
    ) {
        given :
            var device = Neureka.get().backend().get(CLContext.class).platforms[0].devices[0]
            var kernelName = "backend_mm_${M}x${K}x${N}"

            long[] local=   new long[]{ Math.min(16, M), Math.min(16, K) }
            long[] global = new long[]{ M, K }

            var data = (0..(M*K-1)).collect( v-> v + seed )
            var data1 = data.collect( v -> ((v+5)%11)-5 as float )
            var data2 = data.collect( v -> ((v+7)%11)-5 as float )

            Tsr A = Tsr.of( [M,K], data1  )
            Tsr B = Tsr.of( [K,N], data2  )
            Tsr C = Tsr.of( [M,N], 0f )

            var reference = A.matMul(B).value // CPU execution for reference!

            A.to( device )
            B.to( device )
            C.to( device ).setIsVirtual(false)

        expect :
            !device.hasAdHocKernel( kernelName )

        when :
            device.compileAdHocKernel( kernelName, """
    kernel void backend_mm_${M}x${K}x${N}(
        global       float* C,
        global const float* A,
        global const float* B,
        int M, int N, int K
    ){
        local float locA[16][16];
        local float locB[16][16];

        int m = get_global_id(0);
        int k = get_global_id(1);

        int c = ( get_group_id(1) << 4 ) + get_local_id(0);

        float result = 0.0;

        for ( int n = get_local_id(1); n < N; n += 16 )
        {
            locA[ get_local_id(0) ][ get_local_id(1) ] = A[ ( N * m ) + n ];
            locB[ get_local_id(0) ][ get_local_id(1) ] = B[ ( K * n ) + c ];

            barrier(CLK_LOCAL_MEM_FENCE);

            for ( int i = 0; i < 16; i++ )
                result += ( locA[ get_local_id(0) ][i] * locB[ get_local_id(1) ][i] );
           
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        C[ ( K * m ) + k ] = result;
    }
                    """
        )

        then :
            device.hasAdHocKernel( kernelName )

        when :
            device.getAdHocKernel( kernelName )
                    .pass( C ).pass( A ).pass( B )
                    .pass( M ).pass( N ).pass( K )
                    .call( global, local )

        then :
            C.value == reference // GPU should produce the same as CPU!
        and :
            expected == null || C.toString({it.setRowLimit(50)}) == expected

        where :
            seed | M   | K   | N  || expected
            7    | 16  | 16  | 16 || '(16x16):[-8.0, -62.0, -17.0, 39.0, 51.0, 30.0, -13.0, -78.0, 0.0, 34.0, 24.0, -8.0, -62.0, -17.0, 39.0, 51.0, -28.0, 9.0, 24.0, -5.0, -78.0, -8.0, 40.0, 66.0, 59.0, 8.0, -87.0, -28.0, 9.0, 24.0, -5.0, -78.0, 40.0, -8.0, -78.0, -5.0, 24.0, 9.0, -28.0, -87.0, 8.0, 59.0, 66.0, 40.0, -8.0, -78.0, -5.0, 24.0, -13.0, 30.0, ... + 206 more]'
            7    | 32  | 32  | 32 || '(32x32):[160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 122.0, 160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 122.0, 160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 137.0, 165.0, 160.0, 122.0, 51.0, -53.0, -190.0, -305.0, -145.0, -18.0, 76.0, 137.0, 165.0, 160.0, 122.0, 51.0, -53.0, -190.0, ... + 974 more]' // seems to work
            11   | 64  | 64  | 64 || null

    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc compilation works for custom simple row major based matrix multiplication.'(
            int seed, int M, int K, int N, String expected
    ) {
        given :
            var device = Neureka.get().backend().get(CLContext.class).platforms[0].devices[0]
            var kernelName = "simple_backend_mm_${M}x${K}x${N}"

            long[] local=   null //new long[]{ Math.min(16, M), Math.min(16, K) }
            long[] global = new long[]{ M, N }

            var data = (0..(M*K-1)).collect( v-> v + seed )
            var data1 = data.collect( v -> ((v+5)%11)-5 as float )
            var data2 = data.collect( v -> ((v+7)%11)-5 as float )

            Tsr A = Tsr.of( [M,K], data1  )
            Tsr B = Tsr.of( [K,N], data2  )
            Tsr C = Tsr.of( [M,N], 0f )

            var reference = A.matMul(B).value // CPU execution for reference!

            A.to( device )
            B.to( device )
            C.to( device ).setIsVirtual(false)

        expect :
            !device.hasAdHocKernel( kernelName )

        when :
            device.compileAdHocKernel( kernelName, """ 
                                __kernel void $kernelName(                                  
                                      const int M, const int N, const int K,                        
                                      const __global float* A,                                      
                                      const __global float* B,                                      
                                            __global float* C                                       
                               ) {                                                                  
                                   const int m = get_global_id(0); // Row index of C (0..M)         
                                   const int n = get_global_id(1); // Col index of C (0..N)         
                                                                                                    
                                   // Compute a single element (loop over K)                        
                                   float acc = 0.0f;                                                
                                   for ( int k = 0; k < K; k++ )                                    
                                       acc += A[ k + m * K ] * B[ n + k * N ];                      
                                                                                                    
                                   // Store the result                                              
                                   C[ n + m * N ] = acc;                                            
                               }                                                                    
                        """
            )

        then :
            device.hasAdHocKernel( kernelName )

        when :
            device.getAdHocKernel( kernelName )
                    .pass( M ).pass( N ).pass( K )
                    .pass( A ).pass( B ).pass( C )
                    .call( global, local )

        then :
            C.value == reference // GPU should produce the same as CPU!
        and :
            expected == null || C.toString({it.setRowLimit(50)}) == expected

        where : 'Matrix multiplication kernel will work for all dimensions reasonably well!'
            seed | M   | K   | N  || expected
            7    | 2   | 2   | 2  || '(2x2):[8.0, 1.0, 4.0, 1.0]'
            7    | 4   | 4   | 4  || '(4x4):[13.0, 3.0, -7.0, -17.0, -11.0, -5.0, 1.0, 7.0, 31.0, 31.0, 31.0, 31.0, 7.0, 1.0, -5.0, -11.0]'
            7    | 16  | 16  | 16 || '(16x16):[-8.0, -62.0, -17.0, 39.0, 51.0, 30.0, -13.0, -78.0, 0.0, 34.0, 24.0, -8.0, -62.0, -17.0, 39.0, 51.0, -28.0, 9.0, 24.0, -5.0, -78.0, -8.0, 40.0, 66.0, 59.0, 8.0, -87.0, -28.0, 9.0, 24.0, -5.0, -78.0, 40.0, -8.0, -78.0, -5.0, 24.0, 9.0, -28.0, -87.0, 8.0, 59.0, 66.0, 40.0, -8.0, -78.0, -5.0, 24.0, -13.0, 30.0, ... + 206 more]'
            7    | 2   | 4   | 2  || '(2x2):[0.0, -10.0, 16.0, 22.0]'
            7    | 2   | 3   | 2  || '(2x2):[4.0, -5.0, 4.0, 4.0]'
            7    | 17  | 18  | 16 || '(17x16):[-17.0, -68.0, -20.0, 39.0, 54.0, 25.0, -15.0, -77.0, 4.0, 41.0, 34.0, -17.0, -68.0, -20.0, 39.0, 54.0, 15.0, 13.0, -11.0, -101.0, 7.0, 71.0, 80.0, 67.0, -1.0, -113.0, -27.0, 15.0, 13.0, -11.0, -101.0, 7.0, -85.0, -27.0, 9.0, 23.0, -7.0, -81.0, -12.0, 35.0, 71.0, 63.0, 11.0, -85.0, -27.0, 9.0, 23.0, -7.0, 35.0, -12.0, ... + 222 more]'
            7    | 32  | 32  | 32 || '(32x32):[160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 122.0, 160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 122.0, 160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 137.0, 165.0, 160.0, 122.0, 51.0, -53.0, -190.0, -305.0, -145.0, -18.0, 76.0, 137.0, 165.0, 160.0, 122.0, 51.0, -53.0, -190.0, ... + 974 more]' // seems to work
            11   | 16  | 16  | 32 || '(16x32):[-33.0, -23.0, -13.0, -3.0, 40.0, 61.0, 60.0, 37.0, -8.0, -75.0, -43.0, -33.0, -23.0, -13.0, -3.0, 40.0, 61.0, 60.0, 37.0, -8.0, -75.0, -43.0, -33.0, -23.0, -13.0, -3.0, 40.0, 61.0, 60.0, 37.0, -8.0, -75.0, 56.0, 47.0, 38.0, 29.0, -57.0, -44.0, -53.0, -84.0, -16.0, 30.0, 54.0, 56.0, 47.0, 38.0, 29.0, -57.0, -44.0, -53.0, ... + 462 more]'

    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc compilation works for custom column major based tiled matrix multiplication.'(
            int seed, int M, int K, int N, String expected
    ) {
        given :
            def device = Neureka.get().backend().get(CLContext.class).platforms[0].devices[0]
            def kernelName = "fast_columns_major_mm_${M}x${K}x${N}"

            def data = (0..(M*K-1)).collect( v-> v + seed )
            def data1 = data.collect( v -> ((v+5)%11)-5 as float )
            def data2 = data.collect( v -> ((v+7)%11)-5 as float )
        and : 'We create column major based matrices!'
            Tsr A = Tsr.of( [M,K], data1    ).unsafe.toLayout(NDConfiguration.Layout.COLUMN_MAJOR)
            Tsr B = Tsr.of( [K,N], data2    ).unsafe.toLayout(NDConfiguration.Layout.COLUMN_MAJOR)
            Tsr C = Tsr.of( [M,N], 0f ).unsafe.toLayout(NDConfiguration.Layout.COLUMN_MAJOR)

            var reference = A.matMul(B).data // CPU execution for reference!

            A.to( device )
            B.to( device )
            C.to( device ).setIsVirtual(false)

            // Determining optimal tile widths
            var MW = 1
            var KW = 1

            for ( s in [16,8,4,2,1] )
                if ( M % s == 0 ) { MW = s; break }
            for ( s in [8,4,2,1] )
                if ( N % s == 0 && K % s == 0 ) { KW = s; break }

            var NW = KW

            long[] local =  null // This kernel does not have local memory (uses register/private memory instead)
            long[] global = new long[]{ Math.floor(M/MW), Math.floor(N/NW), 1 }

        expect :
            !device.hasAdHocKernel( kernelName )

        when :
            device.compileAdHocKernel( kernelName, """ 
                        #define K ${K}
                        #define N ${N}
                        #define MW ${MW}     // M tile Width
                        #define NW ${NW}     // N tile Width  -- NW & KW should be the same !
                        #define KW ${KW}     // K tile Width
                        #define MT ${Math.floor(M/MW) as int}   // MT is max for 'mt' (M tile count)
                        #define KT ${Math.floor(K/KW) as int}   // KT is max for 'kt' (K tile count)
                        #define floatMW ${MW != 1 ? 'float'+MW : 'float'}
                        #define floatKW ${KW != 1 ? 'float'+KW : 'float'}
                        __kernel void $kernelName(
                                    const __global floatMW* restrict A, 
                                    const __global floatKW* restrict B, 
                                          __global floatMW* C
                        ) {{
                            size_t mt    = get_global_id(0);    //global M-tile id
                            size_t nc    = get_global_id(1);    //global N-tile id
                            size_t batch = get_global_id(2); 
                                
                            float AT[KW][MW]; // sub tiles
                            float BT[NW][KW];
                            float CT[NW][MW];
                            #pragma unroll
                            for ( uint i=0; i<NW*MW; ++i ) // zero CT tile
                                ((float*) CT)[i] = 0.0;
                            for ( uint kt=0; kt<KT; ++kt )  // iterate over K-dim tiles
                            {{
                                #pragma unroll
                                for ( uint k=0; k<KW; ++k )  // every k-element inside K-dim tile
                                    *( (floatMW*) AT[k] ) = A[batch*K*MT + (kt*KW + k)*MT + mt]; // store M-Width floats
                                #pragma unroll
                                for ( uint n=0; n<NW; ++n )  // every n-element inside N-dim tile
                                    *( (floatKW*) BT[n] ) = B[batch*N*KT + (nc*NW + n)*KT + kt]; // store K-Width floats
                                #pragma unroll
                                for ( uint k=0; k<KW; ++k )
                                #pragma unroll
                                for ( uint n=0; n<NW; ++n )  // sub tiles multiplication
                                #pragma unroll
                                for ( uint m=0; m<MW; ++m )
                                    CT[n][m] += AT[k][m] * BT[n][k];
                            }}
                            #pragma unroll
                            for ( uint n = 0; n < NW; ++n )
                                C[ batch * N * MT + ( nc * NW + n ) * MT + mt ] += *( (floatMW*) CT[n] );
                        }}                                                                
                            """
            )

        then :
            device.hasAdHocKernel( kernelName )

        when :
            device.getAdHocKernel( kernelName )
                    .pass( A ).pass( B ).pass( C )
                    .call( global, local )

        then :
            C.data == reference // GPU should produce the same as CPU!
        and :
            C.toString({it.setRowLimit(50)}) == expected

        where : 'This kernel will work for all dimensions well:'
            seed | M   | K   | N  || expected
            7    | 2   | 2   | 2  || '(2x2):[8.0, 1.0, 4.0, 1.0]'
            7    | 4   | 4   | 4  || '(4x4):[13.0, 3.0, -7.0, -17.0, -11.0, -5.0, 1.0, 7.0, 31.0, 31.0, 31.0, 31.0, 7.0, 1.0, -5.0, -11.0]'
            7    | 16  | 16  | 16 || '(16x16):[-8.0, -62.0, -17.0, 39.0, 51.0, 30.0, -13.0, -78.0, 0.0, 34.0, 24.0, -8.0, -62.0, -17.0, 39.0, 51.0, -28.0, 9.0, 24.0, -5.0, -78.0, -8.0, 40.0, 66.0, 59.0, 8.0, -87.0, -28.0, 9.0, 24.0, -5.0, -78.0, 40.0, -8.0, -78.0, -5.0, 24.0, 9.0, -28.0, -87.0, 8.0, 59.0, 66.0, 40.0, -8.0, -78.0, -5.0, 24.0, -13.0, 30.0, ... + 206 more]'
            7    | 2   | 4   | 2  || '(2x2):[0.0, -10.0, 16.0, 22.0]'
            7    | 2   | 3   | 2  || '(2x2):[4.0, -5.0, 4.0, 4.0]'
            7    | 17  | 18  | 16 || '(17x16):[-17.0, -68.0, -20.0, 39.0, 54.0, 25.0, -15.0, -77.0, 4.0, 41.0, 34.0, -17.0, -68.0, -20.0, 39.0, 54.0, 15.0, 13.0, -11.0, -101.0, 7.0, 71.0, 80.0, 67.0, -1.0, -113.0, -27.0, 15.0, 13.0, -11.0, -101.0, 7.0, -85.0, -27.0, 9.0, 23.0, -7.0, -81.0, -12.0, 35.0, 71.0, 63.0, 11.0, -85.0, -27.0, 9.0, 23.0, -7.0, 35.0, -12.0, ... + 222 more]'
            7    | 32  | 32  | 32 || '(32x32):[160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 122.0, 160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 122.0, 160.0, 165.0, 137.0, 76.0, -18.0, -145.0, -305.0, -190.0, -53.0, 51.0, 137.0, 165.0, 160.0, 122.0, 51.0, -53.0, -190.0, -305.0, -145.0, -18.0, 76.0, 137.0, 165.0, 160.0, 122.0, 51.0, -53.0, -190.0, ... + 974 more]' // seems to work
            11   | 16  | 16  | 32 || '(16x32):[-33.0, -23.0, -13.0, -3.0, 40.0, 61.0, 60.0, 37.0, -8.0, -75.0, -43.0, -33.0, -23.0, -13.0, -3.0, 40.0, 61.0, 60.0, 37.0, -8.0, -75.0, -43.0, -33.0, -23.0, -13.0, -3.0, 40.0, 61.0, 60.0, 37.0, -8.0, -75.0, 56.0, 47.0, 38.0, 29.0, -57.0, -44.0, -53.0, -84.0, -16.0, 30.0, 54.0, 56.0, 47.0, 38.0, 29.0, -57.0, -44.0, -53.0, ... + 462 more]'
            5    | 167 | 73  | 88 || '(167x88):[40.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -40.0, -50.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -40.0, -50.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -40.0, -50.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -40.0, -50.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0, -10.0, ... + 14646 more]'

    }



}
