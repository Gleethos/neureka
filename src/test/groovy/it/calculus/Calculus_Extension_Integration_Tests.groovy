package it.calculus

import it.calculus.mocks.CLContext
import neureka.Neureka
import neureka.Tsr
import neureka.autograd.DefaultADAgent
import neureka.backend.api.ExecutionCall
import neureka.backend.api.OperationContext
import neureka.backend.api.operations.OperationBuilder
import neureka.backend.standard.algorithms.GenericAlgorithm
import neureka.backend.standard.implementations.HostImplementation
import neureka.calculus.CalcUtil
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionBuilder
import neureka.devices.Device
import neureka.devices.host.HostCPU
import neureka.devices.opencl.utility.DispatchUtility
import spock.lang.Specification

class Calculus_Extension_Integration_Tests extends Specification
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

            OperationContext oldContext = Neureka.get().context()
            OperationContext testContext = oldContext.clone()

        when:
            def run = testContext.runner()

        then:
            run { testContext == Neureka.get().context() }

        when:
            Tsr t1 = Tsr.of([row_sze, com_sze], -3..8)
            Tsr t2 = Tsr.of([com_sze, col_sze], -7..4)
            run {
                Neureka.get().context()
                    .addOperation(
                        new OperationBuilder()
                                .setFunction('test_function')
                                .setOperator('test_function')
                                .setArity(-1)
                                .setIsIndexer(false)
                                .setIsOperator(false)
                                .setIsDifferentiable(true)
                                .setIsInline(false)
                                .setStringifier(
                                        children -> {
                                            String expression = String.join(", ", children);
                                            if (expression.charAt(0) === '(' && expression.charAt(expression.length() - 1) === ')') {
                                                return "test_function" + expression;
                                            }
                                            return "test_function" + "(" + expression + ")";
                                        }
                                )
                                .build()
                                .setAlgorithm(
                                        GenericAlgorithm.class,
                                        new GenericAlgorithm(null)
                                                .setIsSuitableFor(call -> 1.0f)
                                                .setCanPerformBackwardADFor(call -> true)
                                                .setCanPerformForwardADFor(call -> false)
                                                .setSupplyADAgentFor(
                                                        (Function f, ExecutionCall<? extends Device<?>> call, boolean forward) -> {
                                                            if (forward) throw new IllegalArgumentException("Reshape operation does not support forward-AD!");
                                                            return new DefaultADAgent(null)
                                                                    .setForward((t, derivative) -> new FunctionBuilder( Neureka.get().context() ).build(f.toString(), false).derive(new Tsr[]{derivative}, 0))
                                                                    .setBackward((t, error) -> new FunctionBuilder( Neureka.get().context() ).build(f.toString(), false).derive(new Tsr[]{error}, 0));
                                                        }
                                                )
                                                .setOrchestration((caller, call) -> CalcUtil.executeFor(caller,call,(executionCall, executor) -> null ) )
                                                .setCallPreparation(
                                                        call -> {
                                                            Tsr<?>[] tsrs = call.getTensors();
                                                            Device<?> device = call.getDevice();
                                                            if (tsrs[0] == null) // Creating a new tensor:
                                                            {
                                                                int[] shp = new int[]{tsrs[1].getNDConf().shape()[0], tsrs[2].getNDConf().shape()[1]}
                                                                Tsr output = Tsr.of(shp, 0.0);
                                                                output.setIsVirtual(false);
                                                                device.store(output);
                                                                tsrs[0] = output;
                                                            }
                                                            return call;
                                                        }
                                                )
                                                .setImplementationFor(
                                                        HostCPU.class,
                                                        new HostImplementation(
                                                                (call) -> {
                                                                    Tsr drn = call.getTsrOfType(Number.class, 0)
                                                                    Tsr src1 = call.getTsrOfType(Number.class, 1)
                                                                    Tsr src2 = call.getTsrOfType(Number.class, 2)
                                                                    assert src1.shape(1) == src2.shape(0)

                                                                    //for ( int i=0; i<clContext.getGws(); i++ ) {
                                                                    //    kernel.gemm_template(
                                                                    //            drn.value32(),                     //__global float[] drain,
                                                                    //            drn.getNDConf().asInlineArray(),   //__global int[] drn_conf,
                                                                    //            src1.value32(),                    //const __global float[] src1,
                                                                    //            src1.getNDConf().asInlineArray(),  //__global int[] src1_conf,
                                                                    //            src2.value32(),                    //const __global float[] src2,
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
                                                                },
                                                                3
                                                        )
                                                )
                                )
                )
            }
            Function matMul = run { Function.of("test_function(I[0],I[1])") }


        then :
            matMul.toString() == "test_function(I[0], I[1])"

        when :
            Tsr t3 = run { matMul([t1, t2]) }

        then :
            t3 != null
            //t3.toString() == "..."
        and :
            Neureka.get().context() == oldContext

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
            println( size - expected.inject(1, {a,b -> a*b}) )

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
