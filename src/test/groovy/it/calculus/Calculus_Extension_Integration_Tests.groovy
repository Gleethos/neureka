package it.calculus

import it.calculus.mocks.CLContext
import it.calculus.mocks.GEMMKernelReferenceImplementation
import neureka.Tsr
import neureka.acceleration.Device
import neureka.acceleration.host.execution.HostExecutor
import neureka.autograd.ADAgent
import neureka.calculus.Function
import neureka.calculus.backend.ExecutionCall
import neureka.calculus.backend.implementations.functional.GenericImplementation
import neureka.calculus.backend.operations.OperationContext
import neureka.calculus.backend.operations.OperationType
import neureka.calculus.backend.operations.OperationTypeFactory
import neureka.calculus.frontend.assembly.FunctionBuilder
import spock.lang.Specification

class Calculus_Extension_Integration_Tests extends Specification
{

    def 'GEMM matrix multiplication reference implementation can be set as custom OperationType.'(
            int lws, int gws,
            int max_ts_row,//  = 128, // ts := tile size
            //const u
            int max_ts_col,//  = 128,
            //const u
            int max_wpt_row,// = 8,   // wpt := work per thread
            //const u
            int max_wpt_col, // = 8,
            int row_sze,
            int col_sze
    ) {
        given :
            Tsr t1 = new Tsr([64, 64], -3..8)
            Tsr t2 = new Tsr([64, 64], -7..4)
            def clContext = new CLContext(lws, gws, max_ts_row, max_ts_col, max_wpt_row, max_ts_col, row_sze, col_sze)
            def kernel = new GEMMKernelReferenceImplementation( clContext )

            OperationContext oldContext = OperationContext.instance()
            OperationContext testContext = oldContext.clone()
            OperationContext.setInstance(testContext)
            OperationType type = new OperationTypeFactory()
                    .withFunction('test_function')
                    .withOperator('test_function')
                    .withArity(-1)
                    .setIsIndexer(false)
                    .setIsOperator(false)
                    .setIsDifferentiable(true)
                    .setIsInline(false)
                    .create()
                    .setStringifier(
                            children -> {
                                String expression = String.join( ", ", children );
                                if (expression.charAt(0) === '(' && expression.charAt(expression.length() - 1) === ')') {
                                    return "test_function" + expression;
                                }
                                return "test_function" + "(" + expression + ")";
                            }
                    )
                    .setImplementation(
                            GenericImplementation.class,
                            new GenericImplementation()
                                    .setSuitabilityChecker( call -> 1.0f )
                                    .setBackwardADAnalyzer( call -> true )
                                    .setForwardADAnalyzer(call -> false )
                                    .setADAgentSupplier(
                                            (Function f, ExecutionCall<Device> call, boolean forward ) ->
                                            {
                                                if(forward) throw new IllegalArgumentException("Reshape operation does not support forward-AD!");
                                                return new ADAgent(null)
                                                        .withForward((t, derivative) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{derivative},0))
                                                        .withBackward((t, error) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{error},0));
                                            }
                                    )
                                    .setCallHock( ( caller, call ) -> null )
                                    .setRJAgent( ( call, goDeeperWith ) -> null )
                                    .setDrainInstantiation(
                                            call ->
                                            {
                                                Tsr[] tsrs = call.getTensors();
                                                Device device = call.getDevice();
                                                if ( tsrs[0] == null ) // Creating a new tensor:
                                                {
                                                    int[] shp = tsrs[1].getNDConf().shape();
                                                    Tsr output = new Tsr( shp, 0.0 );
                                                    output.setIsVirtual( false );
                                                    device.add(output);
                                                    tsrs[0] = output;
                                                }
                                                return call;
                                            }
                                    )
                                    .setExecutor(
                                            HostExecutor.class,
                                            new HostExecutor(
                                                    (call) ->
                                                    {
                                                        for ( int i=0; i<gws; i++ ) {
                                                            Tsr drn = call.getTensor(0)
                                                            Tsr src1 = call.getTensor(1)
                                                            Tsr src2 = call.getTensor(2)
                                                            //kernel.gemm_template(
                                                            //        drn.value32(),                     //__global float[] drain,
                                                            //        drn.getNDConf().asInlineArray(), //__global int[] drn_conf,
                                                            //        src1.value32(),                    //const __global float[] src1,
                                                            //        src1.getNDConf().asInlineArray(),//__global int[] src1_conf,
                                                            //        src2.value32(),                    //const __global float[] src2,
                                                            //        src2.getNDConf().asInlineArray(),//__global int[] src2_conf,
                                                            //        //call.getTensor(0).rank(),//int rank, == 2
                                                            //        -1, //const int d,
                                                            //        128, //const u int max_ts_row,//  = 128, // ts := tile size
                                                            //        128, //const u int max_ts_col,//  = 128,
                                                            //        16, //const u int max_ts_com,//  = 16,
                                                            //        8, //const u int max_wpt_row,// = 8,   // wpt := work per thread
                                                            //        8  //const u int max_wpt_col // = 8,
                                                            //)
                                                            clContext.increment()
                                                        }
                                                    },
                                                    3
                                            )
                                    )
                    )

        when :
            Function matMul = Function.create("test_function(I[0],I[1])")

        then :
            matMul.toString() == "test_function(I[0], I[1])"

        when :
            Tsr t3 = matMul([t1, t2])

        then :
            t3 != null
            //t3.toString() == "..."

        where :
            lws | gws | max_ts_row | max_ts_col | max_wpt_row | max_wpt_col | row_sze | col_sze
            8   | 8   | 128        | 128        | 8           | 8           | 64      | 64




    }

}
