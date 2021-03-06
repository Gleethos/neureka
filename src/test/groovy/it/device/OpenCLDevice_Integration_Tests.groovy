package it.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.opencl.OpenCLPlatform
import neureka.devices.opencl.utility.DispatchUtility
import neureka.dtype.DataType
import spock.lang.Specification

class OpenCLDevice_Integration_Tests extends Specification
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
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'An OpenCLDevice will throw an exception when trying to add a tensor whose "data parent" is not outsourced.'()
    {
        given: 'This system supports OpenCL.'
            if (!Neureka.get().canAccessOpenCL()) return
        and : 'The first found OpenCLDevice instance.'
            Device device = Device.find('first')
        and : 'A tensor and a slice tensor of the prior.'
            Tsr t = Tsr.of([4, 3], 2)
            Tsr s = t[1..3, 1..2]

        expect : 'Both tensors share not only the same data but also the same data type.'
            t.data == s.data
            t.dataType == DataType.of( Double.class )
            s.dataType == DataType.of( Double.class )

        when : 'We try to add the slice to the device.'
            device.store(s)

        then : 'An exception is being thrown.'
            def exception = thrown(IllegalStateException)

        and : 'It explains what went wrong.'
            exception.message=="Data parent is not outsourced!"
    }


    def 'The "getValue()" method of an outsourced tensor will return the expected array type.'()
    {
        given : 'This system supports OpenCL'
            if (!Neureka.get().canAccessOpenCL()) return
        and : 'A new tensor.'
            Tsr t = Tsr.ofShape( 1, 2 )

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

    def 'The "getData()" method of an outsourced tensor will return null when outsourced.'()
    {
        given : 'This system supports OpenCL'
            if ( !Neureka.get().canAccessOpenCL() ) return
        and : 'A new tensor belonging to the first found OpenCLDevice instance.'
            Tsr t = Tsr.ofShape( 1, 2 )

        expect : 'The tensor start with having data stored within.'
            t.data != null

        when : 'The tensor is being stored on the device...'
            t.set(Device.find('first'))
        and : 'The tensor value is being fetched...'
            def data = t.getData()

        then : 'The value variable is null.'
            data == null
    }

    def 'Ad hoc compilation produces executable kernel.'() {

        given : 'This system supports OpenCL'
            if ( !Neureka.get().canAccessOpenCL() ) return
            def device = OpenCLPlatform.PLATFORMS()[0].devices[0]
            def someData = Tsr.of( new float[]{ 2, -5, -3, 9, -1 } ).set( device )

        expect :
            !device.hasAdHocKernel( 'dummy_kernel' )

        when :
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

        then :
            device.hasAdHocKernel( 'dummy_kernel' )

        and :
            device._adhocKernels['dummy_kernel'].source == """
                __kernel void dummy_kernel (
                        __global float* output,
                        __global float* input,
                        float value 
                    ) { 
                        unsigned int i = get_global_id( 0 );
                        output[i] = input[i] + value; 
                    }
                """

        when :
            device.getAdHocKernel( 'dummy_kernel' )
                    .passRaw( someData )
                    .passRaw( someData )
                    .pass( -4f )
                    .call( someData.size() )

        then :
            someData.toString() == "(5):[-2.0, -9.0, -7.0, 5.0, -5.0]"

    }


    def 'Ad hoc compilation works for matrix multiplication.'(
           int regSize, int locSize, def M, def K, def N, String expected
    ) {

        given : 'This system supports OpenCL'
            if ( !Neureka.get().canAccessOpenCL() ) return
            def device = OpenCLPlatform.PLATFORMS()[0].devices[0]
            def kernelName = "dummy_mm_${M}x${K}x${N}"
            def params = DispatchUtility.findBestParams(locSize, regSize, K, M, N)

            //{max_ts_row, max_ts_col, max_ts_com, max_wpt_row, max_wpt_col}
            def TSM  = params[0] //4   //= 128
            def TSN  = params[1] //4   //= 128
            def TSK  = params[2] //2   //= 16
            def WPTM = params[3] // 2  //=  8
            def WPTN = params[4] // 2  //=  8

            long[] local=   new long[]{ TSM/WPTM, TSN/WPTN }
            long[] global = new long[]{ (M/WPTM), (N/WPTN) }
            //println('TSM:'+TSM+' - TSN:'+TSN+' - TSK:'+TSK+' - WPTM'+WPTM+' - WPTN:'+WPTN+' |'+local+'|'+global)

            Tsr A = Tsr.of( [M,K], 0 )
            Tsr B = Tsr.of( [K,N], 0 )
            Tsr C = Tsr.of( [M,N], 0 )
            A[0..M-1,0..K-1] = Tsr.of([M,K], 3..1)
            B[0..K-1,0..N-1] = Tsr.of([K,N], -5..0)
            //println(A)
            println(A)
            println(B)
            A.set( device )
            B.set( device )
            C.set( device )
            //println(C)

        expect :
            !device.hasAdHocKernel( kernelName )

        when :
            device.compileAdHocKernel( kernelName, """
                    #define TSM ${TSM}                      // The tile-size in dimension M
                    #define TSN ${TSN}                      // The tile-size in dimension N
                    #define TSK ${TSK}                   // The tile-size in dimension K
                    #define WPTM ${WPTM}                  // The work-per-thread in dimension M
                    #define WPTN ${WPTN}                 // The work-per-thread in dimension N
                    #define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M
                    #define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N
                    #define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
                    #define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

                    // Use 2D register blocking (further increase in work per thread)
                    __kernel void ${kernelName}(
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
                    .passRaw( A ).passRaw( B ).passRaw( C )
                    .call( global, local )

        then :
            C.toString() == expected

        where :
             regSize | locSize | M  | K  | N  || expected
               2**2  |  4**2   | 4  | 4  | 4  || '(4x4):[-35.0, -26.0, -29.0, -20.0, -30.0, -22.0, -20.0, -12.0, -19.0, -12.0, -23.0, -16.0, -35.0, -26.0, -29.0, -20.0]'
                16   |  32     | 16 | 16 | 16 || '(16x16):[-115.0, -82.0, -109.0, -76.0, -73.0, -40.0, -115.0, -82.0, -109.0, -76.0, -73.0, -40.0, -115.0, -82.0, -109.0, -76.0, -110.0, -78.0, -76.0, -44.0, -102.0, -70.0, -110.0, -78.0, -76.0, -44.0, -102.0, -70.0, -110.0, -78.0, -76.0, -44.0, -75.0, -44.0, -103.0, -72.0, -101.0, -70.0, -75.0, -44.0, -103.0, -72.0, -101.0, -70.0, -75.0, -44.0, -103.0, -72.0, -115.0, -82.0, ... + 206 more]'
               // 16   |  32     | 8  | 8  | 8  || '?'
               //2**2  |  4**2   | 4  | 6  | 5  || '?'// BROKEN

    }



}
