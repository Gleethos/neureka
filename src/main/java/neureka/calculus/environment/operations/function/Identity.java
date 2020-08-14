package neureka.calculus.environment.operations.function;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.*;

public class Identity extends OperationType
{

    public Identity()
    {
        super("identity", "idy" , 1, false, false, false, true, true);

        setStringifier(
                children -> {
                    String expression = String.join( ", ", children );
                    if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                        return "idy" + expression;
                    }
                    return "idy" + "(" + expression + ")";
                }
        );

        DefaultOperatorCreator<TertiaryNDXConsumer> activationCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                    else return (t0Idx, t1Idx, t2Idx) -> 1;
                };

        Activation typeImplementation = new Activation(
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
                ( call, goDeeperWith ) -> null,
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[1].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual(false);
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
        );
        setImplementation(
                Activation.class,
                typeImplementation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor(0),
                                                                        start, end,
                                                                        activationCreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                    // Drain tensor needs to be 'actual'! :
                                    if(!call.getTensor(offset + 1).isVirtual()) call.getTensor(offset).setIsVirtual(false);
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                typeImplementation.getKernelSource(), // kernelSource
                                "output = input;\n", // activationSource
                                "output = input;\n", //differentiationSource
                                this // OperationType
                        )
                )
        );

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarizationCreator =
                (inputs, value, d) -> {
                    if (d < 0) return t1Idx -> value;
                    else return t1Idx -> value;
                };
        Scalarization scalarization = new Scalarization(
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
                ( call, goDeeperWith ) -> null,
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[1].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual(false);
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
        );
        setImplementation(Scalarization.class,
                scalarization.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call  -> {
                                    double value = call.getTensor(0).value64(2);
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        (start, end) ->
                                                                Scalarization.scalarize(
                                                                        call.getTensor(0), start, end,
                                                                        scalarizationCreator.create(
                                                                                call.getTensors(), value, call.getDerivativeIndex()
                                                                        )
                                                                )
                                                );
                                },
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    Tsr t = call.getTensor(0);
                                    int gwz = t.size();
                                    call.getDevice().getKernel(call)
                                            .pass(t)
                                            .pass(t)
                                            .pass((float)call.getTensor(1).value64(0))
                                            .pass(t.rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                scalarization.getKernelSource(), // kernelSource
                                "output = value;\n",
                                "output = value;\n",
                                this // OperationType
                        )
                )
        );


    }

}
