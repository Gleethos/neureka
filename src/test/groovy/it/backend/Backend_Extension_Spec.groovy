package it.backend

import it.backend.mocks.CLContext
import neureka.Neureka
import neureka.Tsr
import neureka.autograd.ADAgent
import neureka.backend.api.BackendContext
import neureka.backend.api.DeviceAlgorithm
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.backend.api.algorithms.fun.AutoDiffMode
import neureka.backend.api.algorithms.fun.Result
import neureka.backend.api.algorithms.fun.SuitabilityPredicate
import neureka.backend.standard.implementations.CPUImplementation
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionParser
import neureka.calculus.internal.CalcUtil
import neureka.devices.Device
import neureka.devices.host.CPU
import spock.lang.Specification
import testutility.opencl.DispatchUtility

class Backend_Extension_Spec extends Specification
{

    def 'GEMM matrix multiplication reference implementation can be set as custom OperationType and works as expected.'(
            int lws,
            int rws,
            int com_sze,
            int row_sze,
            int col_sze
    ) {
        given :
            //def clContext = new CLContext(lws, rws, com_sze, row_sze, col_sze)
            //def kernel = new GEMMKernelReferenceImplementation( clContext )

            BackendContext oldContext = Neureka.get().backend()
            BackendContext testContext = oldContext.clone()

        when:
            def run = testContext.runner()

        then:
            run { testContext == Neureka.get().backend() }

        when:
            Tsr t1 = Tsr.of([row_sze, com_sze], -3d..8d)
            Tsr t2 = Tsr.of([com_sze, col_sze], -7d..4d)
            run {
                Neureka.get().backend()
                    .addOperation(
                            Operation
                                .builder()
                                .setIdentifier('test_function')
                                .setOperator('test_function')
                                .setArity(-1)
                                .setIsIndexer(false)
                                .setIsOperator(false)
                                .setIsDifferentiable(true)
                                .setIsInline(false)
                                .setStringifier(
                                        children -> {
                                            String expression = String.join(", ", children);
                                            if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                                                return "test_function" + expression;
                                            }
                                            return "test_function" + "(" + expression + ")";
                                        }
                                )
                                .build()
                                .setAlgorithm(
                                        DeviceAlgorithm.withName(null)
                                            .setIsSuitableFor(call -> SuitabilityPredicate.GOOD  )
                                            .setAutogradModeFor(call -> AutoDiffMode.BACKWARD_ONLY )
                                            .setExecution( (caller, call) ->
                                                Result.of(CalcUtil.defaultRecursiveExecution(caller, call))
                                                    .withAutoDiff((Function f, ExecutionCall<? extends Device<?>> adCall, boolean forward) -> {
                                                        if (forward) throw new IllegalArgumentException("Reshape operation does not support forward-AD!");
                                                        return ADAgent.withAD((t, error) -> new FunctionParser( Neureka.get().backend() ).parse(f.toString(), false).derive(new Tsr[]{error}, 0));
                                                    })
                                            )
                                            .setCallPreparation(
                                                    call -> {
                                                         Device<?> device = call.getDevice();
                                                        if ( call.input( 0 ) == null ) // Creating a new tensor:
                                                        {
                                                            int[] shp = new int[]{call.input( 1 ).getNDConf().shape(0), call.input( 2 ).getNDConf().shape(1)}
                                                            Tsr output = Tsr.of(shp, 0.0);
                                                            output.setIsVirtual(false);
                                                            device.store( output );
                                                            call.setInput( 0, output );
                                                        }
                                                        return call;
                                                    }
                                            )
                                            .setImplementationFor(
                                                    CPU.class,
                                                    CPUImplementation
                                                        .withArity(3)
                                                        .andImplementation(
                                                            (call) -> {
                                                                Tsr drn = call.input(Number.class, 0)
                                                                Tsr src1 = call.input(Number.class, 1)
                                                                Tsr src2 = call.input(Number.class, 2)
                                                                assert src1.shape(1) == src2.shape(0)

                                                                //for ( int i=0; i<clContext.getGws(); i++ ) {
                                                                //    kernel.gemm_template(
                                                                //            drn.getDataAs( float[].class ),                     //__global float[] drain,
                                                                //            drn.getNDConf().asInlineArray(),   //__global int[] drn_conf,
                                                                //            src1.getDataAs( float[].class ),                    //const __global float[] src1,
                                                                //            src1.getNDConf().asInlineArray(),  //__global int[] src1_conf,
                                                                //            src2.getDataAs( float[].class ),                    //const __global float[] src2,
                                                                //            src2.getNDConf().asInlineArray(),  //__global int[] src2_conf,
                                                                //            //call.getTsrOfType( Number.class, 0).rank(),//int rank, == 2
                                                                //            //-1, //const int d,
                                                                //            clContext.getMaxTSRow(),//128, //const u int max_ts_row,//  = 128, // ts := tile size
                                                                //            clContext.getMaxTSCol(),//128, //const u int max_ts_col,//  = 128,
                                                                //            clContext.getMaxTSCom(),//16, //const u int max_ts_com,//  = 16,
                                                                //            clContext.getMaxWPTRow(),//8, //const u int max_wpt_row,// = 8,   // wpt := work per thread
                                                                //            clContext.getMaxWPTCol()//8  //const u int max_wpt_col // = 8,
                                                                //    )
                                                                //    clContext.increment()
                                                                //}
                                                            }
                                                        )
                                            )
                                            .buildFunAlgorithm()
                                )
                )
            }
            Function testFun = run { Function.of("test_function(I[0],I[1])") }


        then :
            testFun.toString() == "test_function(I[0], I[1])"

        when :
            Tsr t3 = run { testFun([t1, t2]) }

        then :
            t3 != null
            //t3.toString() == "..."
        and :
            Neureka.get().backend() == oldContext

        where :
            lws | rws | com_sze | row_sze | col_sze
            320 | 32  | 80      | 640     | 640
            32  | 16  | 8       | 64      | 64


    }

    def 'Test context mock for opencl reference implementations.'(
            int lws, int rws,
            int com_sze, int row_sze, int col_sze,
            int ts_com, int ts_row, int ts_col, int wpt_row, int wpt_col
    ) {
        given :
            def clContext = new CLContext(lws, rws, com_sze, row_sze, col_sze)

        expect :
            clContext.getMaxWPTCol()==wpt_col
            clContext.getMaxWPTRow()==wpt_row
            clContext.getMaxTSCom()==ts_com
            clContext.getMaxTSCol()==ts_col
            clContext.getMaxTSRow()==ts_row

        where :
            lws | rws | com_sze | row_sze | col_sze || ts_com | ts_row | ts_col | wpt_row | wpt_col
            32  | 16  | 8       | 64      | 64      || 4      | 4      | 4      |   4     | 4
            320 | 32  | 80      | 640     | 640     || 16     | 16     | 16     |   4     | 4
            738 | 84  | 345     | 848     | 738     || 23     | 16     | 18     |   8     | 9
    }



    def 'Tile parsing for kernel parameter calculation yields expected tile dimensions.'(
            int size, int[] shape, List<Integer> expected
    ){

        when :
            int[] result = DispatchUtility.parseTile( size, shape as int[] )

        then :
            result == expected as int[]
            //println( size - expected.inject(1, {a,b -> a*b}) )

        where :
            size  | shape                 || expected
            800   | [432, 93, 352, 193]   || [8, 31, 4, 1]
            1800  | [432, 903, 3520, 193] || [108, 3, 8, 1]
            800   | [422, 293]            || [2, 293]
            600   | [993]                 || [331]
            200   | [252, 143]            || [12, 13]
            100   | [100, 100]            || [10, 5]
            100   | [400, 100]            || [8, 5]
            255   | [470, 652]            || [47, 4]
            255   | [7849, 4782]          || [47, 6]
           128**2 | [256, 256]            || [128, 128]
             8**2 | [128, 128]            || [8, 8]
            4**2  | [4,4]                 || [4,4]
    }



}
