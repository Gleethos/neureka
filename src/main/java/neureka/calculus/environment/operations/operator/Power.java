package neureka.calculus.environment.operations.operator;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;
import neureka.calculus.environment.implementations.*;

public class Power extends OperationType
{

    private final static DefaultOperatorCreator<TertiaryNDXConsumer> _creator = (inputs, d)->
    {
        double[] t1_val = inputs[1].value64();
        double[] t2_val = inputs[2].value64();
        if (d < 0) {
            return (t0Idx, t1Idx, t2Idx) -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], t2_val[inputs[2].i_of_idx(t2Idx)]);
        } else {
            return (t0Idx, t1Idx, t2Idx) -> {
                if (d == 0) {
                    return t2_val[inputs[2].i_of_idx(t2Idx)]
                            * Math.pow(
                            t1_val[inputs[1].i_of_idx(t1Idx)],
                            t2_val[inputs[2].i_of_idx(t2Idx)] - 1
                    );
                } else {
                    return Math.pow(
                            t1_val[inputs[1].i_of_idx(t1Idx)],
                            t2_val[inputs[2].i_of_idx(t2Idx)]
                    ) * Math.log(t1_val[inputs[1].i_of_idx(t1Idx)]);
                }
            };
        }
    };

    public Power()
    {
        super("power", "^", -1, true, false, false, false, false);

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" ^ ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<PrimaryNDXConsumer> operationCreator = (inputs, d)->
        {
            double[] t1_val = inputs[1].value64();
            double[] t2_val = inputs[2].value64();
            if (d < 0) {
                return t1Idx -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], t2_val[inputs[2].i_of_idx(t1Idx)]);
            } else {
                return t1Idx -> {
                    if (d == 0) {
                        return t2_val[inputs[2].i_of_idx(t1Idx)]
                                * Math.pow(
                                t1_val[inputs[1].i_of_idx(t1Idx)],
                                t2_val[inputs[2].i_of_idx(t1Idx)] - 1
                        );
                    } else {
                        return Math.pow(
                                t1_val[inputs[1].i_of_idx(t1Idx)],
                                t2_val[inputs[2].i_of_idx(t1Idx)]
                        ) * Math.log(t1_val[inputs[1].i_of_idx(t1Idx)]);
                    }
                };
            }
        };

        OperationTypeImplementation.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            OperationType type = call.getType();

            Tsr alternative = null;
            if ( tsrs.length > 3 )
            {
                if ( d < 0 ) {
                    Tsr[] reduction = new Tsr[]{tsrs[0], tsrs[1], tsrs[2]};
                    alternative = goDeeperWith.apply(
                            call.withNew( reduction )
                    );
                    tsrs[0] = reduction[0];

                    reduction = AbstractOperationTypeImplementation.Utility._offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            call.withNew( reduction )
                            );
                    tsrs[0] = reduction[0];
                } else {

                    if ( d==0 ) {
                        Tsr[] reduction = AbstractOperationTypeImplementation.Utility._subset(tsrs, 1,  2, tsrs.length-2);
                        reduction[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationType.instance("*") )
                        );
                        Tsr exp = reduction[0];
                        reduction = new Tsr[]{tsrs[0], tsrs[1], exp};
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, 0, type )
                        );
                        tsrs[0] = reduction[0];
                        exp.delete();
                    } else {
                        Tsr[] reduction = AbstractOperationTypeImplementation.Utility._subset(tsrs, 1,  2, tsrs.length-2);

                        reduction[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, d-1, OperationType.instance("*") )
                        );
                        Tsr inner = reduction[0];

                        reduction = new Tsr[]{Tsr.Create.newTsrLike(tsrs[1]), inner, tsrs[d]};
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationType.instance("*") )
                        );
                        Tsr exp = reduction[0];

                        reduction = new Tsr[]{tsrs[0], tsrs[1], exp};
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, 1, type )
                        );
                        tsrs[0] = reduction[0];

                        inner.delete();
                        exp.delete();
                    }
                }
                return alternative;
            } else {
                return alternative;
            }


        };

        Operation operation = new Operation(
        ).setADAnalyzer(
                call -> true
        ).setCallHock(
                (caller, call) -> null
        ).setRJAgent(
                rja
        ).setDrainInstantiation(
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

        setImplementation(Operation.class,
                operation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Operation.operate (
                                                                        call.getTensor(0),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        operationCreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                operation.getKernelSource(), // kernelSource
                                "output = pow(input1, input2);",
                                "if(d==0) {                                    \n" +
                                        "    output = input2 * pow(input1, input2-1.0f);  \n" +
                                        "} else {                                         \n" +
                                        "    output = pow(input1, input2) * log(input1);  \n" +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast(
        ).setADAnalyzer(
                call -> true
        ).setCallHock(
                (caller, call) -> null
        ).setRJAgent(
            rja
        ).setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[1].getNDConf().shape();
                        //int[] shp = (type.identifier().endsWith("x"))
                        //        ? Tsr.Utility.Indexing.shpOfCon(tsrs[1].getNDConf().shape(), tsrs[2].getNDConf().shape())
                        //        : tsrs[1].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual(false);
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
        );

        setImplementation(
                Broadcast.class,
                broadcast.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor(0), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        _creator.create(call.getTensors(), call.getDerivativeIndex())
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
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                broadcast.getKernelSource(), // kernelSource
                                "value += pow(src1, src2);",
                                "if(d==0){\n" +
                                        "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                                        "} else {\n" +
                                        "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarCreator =
                ( inputs, value, d )->{
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) {
                        return t1Idx -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value);
                    } else {
                        if(d==0){
                            return t1Idx -> value*Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value-1);
                        } else {
                            return t1Idx -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value)*Math.log(value);
                        }
                    }
                };


        Scalarization scalarization = new Scalarization(
                call -> true,
                (caller, call) -> null,
                rja,
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[1].getNDConf().shape();
                        //int[] shp = (type.identifier().endsWith("x"))
                        //        ? Tsr.Utility.Indexing.shpOfCon(tsrs[1].getNDConf().shape(), tsrs[2].getNDConf().shape())
                        //        : tsrs[1].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual(false);
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
        )    ;

        setImplementation(
                Scalarization.class,
                scalarization.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call -> {
                                    double value = call.getTensor(0).value64(2);
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTensor(0).size(),
                                                    ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTensor(0),
                                                                    start, end,
                                                                    scalarCreator.create(call.getTensors(), value, -1)
                                                            )
                                            );
                                },
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(2).isVirtual() || call.getTensor(2).size() == 1)?1:0;
                                    int gwz = call.getTensor(0).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(0))
                                            .pass(call.getTensor(0))
                                            .pass((float)call.getTensor(1+offset).value64(0))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                scalarization.getKernelSource(), // kernelSource
                                "output = pow(input1, value);",
                                "if ( d==0 ) {                                     \n" +
                                        "    output = value * pow(input1, value-(float)1 );   \n" +
                                        "} else {                                             \n" +
                                        "    output = pow(input1, value) * log(value);        \n" +
                                        "}",
                                this // OperationType
                        )
                )
        );




        //__________________________
        // RELATED OPERATION TYPES :

        new OperationType("inv_power_left", ((char)171)+"^", 3, true, false, false, false, false);
        new OperationType("inv_power_right", "^" + ((char) 187), 3, true, false, false, false, false);

        // Convolution:

        new OperationType(
                "power", "p", 2, true, false, true, false, false
        ).setImplementation(
                Convolution.class,
                new Convolution()
                    .setADAnalyzer(
                            call -> {
                                if ( call.getType().supports(Convolution.class) ) return false;
                                if ( call.getType().identifier().equals(",") ) return false; //Reshape
                                Tsr last = null;
                                for ( Tsr t : call.getTensors() ) {
                                    if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                    last = t; // Note: shapes are cached!
                                }
                                return true;
                            }
                    ).setCallHock(
                            ( caller, call ) -> null
                    ).setRJAgent(
                            ( call, goDeeperWith ) -> null
                    ).setDrainInstantiation(
                            call -> {
                                Tsr[] tsrs = call.getTensors();
                                int offset = ( tsrs[0] == null ) ? 1 : 0;
                                return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                            }
                    )
        ).setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" p ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        new OperationType("", ((char) 171) + "p", 3, true, false, true, false, false);
        new OperationType("", "p" + ((char) 187), 3, true, false, true, false, false);




    }

}
