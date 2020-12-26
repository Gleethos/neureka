package neureka.backend.standard.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.devices.Device;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.Operation;
import neureka.backend.api.algorithms.Algorithm;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Product extends AbstractOperation {


    public Product()
    {
        super (
                "prodJs",
                "prodJs",
                1,
                false,
                true,
                true,
                false
        );

        setStringifier(
                children -> {
                    String expression = String.join( ", ", children );
                    if (expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')') {
                        return "prodJs" + expression;
                    }
                    return "prodJs" + "(" + expression + ")";
                }
        );

        Algorithm.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            Operation type = call.getOperation();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if ( d < 0 ) {
                    Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];

                    reduction = Utility.offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];
                } else {
                    Tsr[] reduction = Utility.without(tsrs, 1+d);
                    if ( reduction.length > 2 ) {
                        reduction[ 0 ] = ( reduction[ 0 ] == null ) ? Tsr.Create.newTsrLike(tsrs[ 1 ]) : reduction[ 0 ];
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, Operation.instance("*") )
                        );
                        tsrs[ 0 ] = reduction[ 0 ];
                    } else tsrs[ 0 ] = reduction[ 1 ];
                }
                return alternative;
            } else {
                return alternative;
            }
        };


        //________________
        // BROADCASTING :

        DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
                ( inputs, d ) ->
                {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] * t2_val[t2Idx.i()];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if (d == 0) return t2_val[t2Idx.i()];
                            else return t1_val[ t1Idx.i() ];
                        };
                    }
                };

        DefaultOperatorCreator<TertiaryNDXConsumer> _creatorX =
                ( inputs, d ) ->
                {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].i_of_idx( t1Idx )] * t2_val[inputs[ 2 ].i_of_idx(t2Idx)];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if (d == 0) return t2_val[inputs[ 2 ].i_of_idx(t2Idx)];
                            else return t1_val[inputs[ 1 ].i_of_idx( t1Idx )];
                        };
                    }
                };

        Broadcast operationAlgorithm = new Broadcast()
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if ( ctxDerivative != null ) {
                                return new DefaultADAgent( ctxDerivative )
                                    .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                    .setBackward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr<?> deriv = f.derive( inputs, d );
                            return new DefaultADAgent( deriv )
                                    .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                    .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, deriv}) );
                        }
                    }
                )
                .setRJAgent( rja )
                .build();

        setAlgorithm(
                Broadcast.class,
                operationAlgorithm.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor( 0 ),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        _creatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        :  ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor( 0 ),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        _creator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = ( call.getTensor( 0 ) != null ) ? 0 : 1;
                                    int gwz = ( call.getTensor( 0 ) != null ) ? call.getTensor( 0 ).size() : call.getTensor( 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTensor( offset ) )
                                            .pass( call.getTensor( offset + 1 ) )
                                            .pass( call.getTensor( offset + 2 ) )
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                operationAlgorithm.getKernelSource(), // kernelSource
                                "value = src1 * src2;\n",
                                "value += handle * drain;\n",
                                this // OperationType
                        )
                )
        );

        //______________
        // ACTIVATION :

        DefaultOperatorCreator<TertiaryNDIConsumer> activationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                    else return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                };

        DefaultOperatorCreator<TertiaryNDXConsumer> activationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].i_of_idx( t1Idx )];
                    else return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].i_of_idx( t1Idx )];
                };

        Activation activation = new Activation()
        .setBackwardADAnalyzer( call -> true )
        .setForwardADAnalyzer(
                call -> true
        ).setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if ( ctxDerivative != null ) {
                            return new DefaultADAgent( ctxDerivative )
                                .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                .setBackward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if ( forward )
                        {
                            Tsr deriv = f.derive( inputs, d );
                            return new DefaultADAgent( deriv )
                                    .setForward( (t, derivative ) -> mul.call(new Tsr[]{derivative, deriv}) )
                                    .setBackward( (t, derivative ) -> mul.call(new Tsr[]{derivative, deriv}) );
                        }
                        else
                        {
                            if ( this.supports(Convolution.class) )
                            {
                                Function invX = FunctionBuilder.build(
                                        "I[ 0 ]" + getOperator() + ">>I[ 1 ]" + getOperator() + ">>I[ 2 ]",
                                        false
                                );
                                Tsr deriv = f.derive( inputs, d );
                                return new DefaultADAgent( deriv )
                                        .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                        .setBackward( (t, error) -> invX.call(new Tsr[]{error, deriv, new Tsr(t.getPayload().shape(), 0)}) );
                            }
                            else
                            {
                                Tsr deriv = f.derive( inputs, d );
                                return new DefaultADAgent( deriv )
                                        .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                        .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, deriv}) );
                            }
                        }
                    }
        )
        .setCallHook( (caller, call ) -> null )
        .setRJAgent( rja )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[ 1 ].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual( false );
                        try {
                            device.store(output);
                        } catch( Exception e ) {
                            e.printStackTrace();
                        }
                        tsrs[ 0 ] = output;
                    }
                    return call;
                }
        )
        .build();

        setAlgorithm(Activation.class,
                activation.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor( 0 ),
                                                                        start, end,
                                                                        activationXCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor( 0 ), call.getTensor( 1 ),
                                                                        start, end,
                                                                        activationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = (call.getTensor( 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTensor( 0 ) != null) ? call.getTensor( 0 ).size() : call.getTensor( 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTensor( offset ) )
                                            .pass( call.getTensor( offset + 1 ) )
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                activation.getKernelSource(), // kernelSource
                                "output = input;",
                                "output = 1;",
                                this // OperationType
                        )
                )
        );




    }



    @Override
    public double calculate(double[] inputs, int j, int d, List<Function> src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double prod = 1;
            boolean nothingDone = true;
            for ( int Ii = 0; Ii < inputs.length; Ii++ ) {
                prod *= src.get( 0 ).call( inputs, Ii );
                nothingDone = false;
            }
            if ( nothingDone ) return src.get( 0 ).call( inputs, j );
            return prod;
        } else {
            double u, ud, v, vd;
            u = src.get( 0 ).call( inputs, 0 );
            ud = src.get( 0 ).derive(inputs, d, 0);
            for (int ji = 1; ji < inputs.length; ji++) {
                v = src.get( 0 ).call( inputs, ji );
                vd = src.get( 0 ).derive( inputs, d, ji );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }

    @Contract(pure = true)
    public static double calculate(double[] inputs, int d, List<Function> src ) {
        if ( d < 0 ) {
            double prod = 1;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                prod *= src.get( 0 ).call(inputs, i);
                nothingDone = false;
            }
            if ( nothingDone ) return src.get( 0 ).call( inputs );
            return prod;
        } else {
            double u, ud, v, vd;
            u = src.get( 0 ).call(inputs, 0);
            ud = src.get( 0 ).derive(inputs, d, 0);
            for ( int j = 1; j < inputs.length; j++ ) {
                v = src.get( 0 ).call( inputs, j );
                vd = src.get( 0 ).derive( inputs, d, j );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }


}
