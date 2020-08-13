package neureka.calculus.environment.operations.convolution;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;
import neureka.calculus.environment.implementations.AbstractOperationTypeImplementation;
import neureka.calculus.environment.implementations.Convolution;

public class XMultiplication extends OperationType
{

    public XMultiplication(){

        super(
                "multiply",
                "x",
                2,
                true,
                false,
                true,
                false,
                false
        );

        OperationTypeImplementation.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            OperationType type = call.getType();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if (d < 0) {
                    Tsr[] reduction = new Tsr[]{tsrs[0], tsrs[1], tsrs[2]};
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];

                    reduction = AbstractOperationTypeImplementation.Utility._offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];
                }
                return alternative;
            } else {
                if ( call.getType().identifier().equals("x") ) {
                    if (d >= 0) {
                        if (d == 0) tsrs[0] = tsrs[2];
                        else tsrs[0] = tsrs[1];
                        return tsrs[0];
                    } else {
                        call.mutateArguments( t -> new Tsr[]{t[0], t[1], t[2]} );
                    }
                } else if ( call.getType().identifier().equals("x"+ ((char) 187)) ) {
                    call.mutateArguments( t -> new Tsr[]{t[2], t[1], t[0]} );
                }
                return alternative;
            }
        };

        DefaultOperatorCreator<TertiaryNDXConsumer> convolutionCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if (d < 0) {
                        return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                    } else {
                        return (t0Idx, t1Idx, t2Idx) -> {
                            if (d == 0) return t2_val[inputs[2].i_of_idx(t2Idx)];
                            else return t1_val[inputs[1].i_of_idx(t1Idx)];
                        };
                    }
                };

        Convolution convolution = new Convolution(
                call -> {
                    if ( call.getType().supports(Convolution.class) ) return false;
                    if ( call.getType().identifier().equals(",") ) return false; //Reshape
                    Tsr last = null;
                    for ( Tsr t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                },
                rja,
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = Tsr.Utility.Indexing.shpOfCon(tsrs[1].getNDConf().shape(), tsrs[2].getNDConf().shape());
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual(false);
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
        )    ;

        setImplementation(
                Convolution.class,
                convolution
                        .setExecutor(
                                HostExecutor.class,
                                new HostExecutor(
                                        call ->
                                                call.getDevice().getExecutor()
                                                        .threaded (
                                                                call.getTensor(0).size(),
                                                                ( start, end ) ->
                                                                        Convolution.convolve (
                                                                                call.getTensor(0), call.getTensor(1), call.getTensor(2),
                                                                                call.getDerivativeIndex(), start, end,
                                                                                convolutionCreator.create(
                                                                                        call.getTensors(),
                                                                                        -1//call.getDerivativeIndex()
                                                                                )
                                                                        )
                                                        ),
                                        3
                                )
                        ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())//call.getDerivativeIndex()
                                            .call(gwz);
                                },
                                3,
                                convolution.getKernelSource(), // kernelSource
                                "value = src1 * src2;\n",
                                "value += handle * drain;\n",
                                this // OperationType
                        )
                )
        );
        new OperationType(
                "inv_convolve_mul_left", ((char) 171) + "x", 3, true, false, true, false, false
        ).setImplementation(Convolution.class, convolution);

        new OperationType(
                "inv_convolve_mul_right", "x" + ((char) 187), 3, true, false, true, false, false
        ).setImplementation(Convolution.class, convolution);


    }



}
