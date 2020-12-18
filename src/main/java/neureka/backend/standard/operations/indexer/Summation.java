package neureka.backend.standard.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.devices.Device;
import neureka.devices.host.execution.HostExecutor;
import neureka.devices.opencl.execution.CLExecutor;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.backend.standard.implementations.Activation;
import neureka.backend.standard.implementations.Broadcast;
import neureka.backend.standard.implementations.Convolution;
import neureka.backend.api.operations.AbstractOperationType;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.OperationType;
import neureka.backend.api.implementations.OperationTypeImplementation;
import neureka.calculus.frontend.assembly.FunctionBuilder;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Summation extends AbstractOperationType
{

    public Summation()
    {
        super (
                "sumJs",
                "sumJs",
                1,
                false,
                true,
                true,
                false
        );

        setStringifier(
                children ->
                {
                    String expression = String.join( ", ", children );
                    if (expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')') {
                        return "sumJs" + expression;
                    }
                    return "sumJs" + "(" + expression + ")";
                }
        );

        OperationTypeImplementation.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            OperationType type = call.getOperation();

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
                    tsrs[ 0 ] = Tsr.Create.newTsrLike(tsrs[ 1 ]).setValue(1.0f);
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
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] + t2_val[t2Idx.i()];
                    else return ( t0Idx, t1Idx, t2Idx ) -> 1.0;
                };

        DefaultOperatorCreator<TertiaryNDXConsumer> _creatorX =
                ( inputs, d ) ->
                {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].i_of_idx( t1Idx )] + t2_val[inputs[ 2 ].i_of_idx(t2Idx)];
                    else return ( t0Idx, t1Idx, t2Idx ) -> 1.0;
                };

        Broadcast typeImplementation = new Broadcast()
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>)call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if ( ctxDerivative != null ) {
                            return new DefaultADAgent( ctxDerivative )
                                    .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                    .setBackward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr deriv = f.derive( inputs, d );
                            return new DefaultADAgent( deriv )
                                    .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                    .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, deriv}) );
                        }
                    }
                )
                .setRJAgent( rja )
                .build();


        setImplementation (
                Broadcast.class,
                typeImplementation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
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
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor( 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTensor( 0 ) != null) ? call.getTensor( 0 ).size() : call.getTensor( 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTensor( offset ) )
                                            .pass( call.getTensor( offset + 1 ) )
                                            .pass( call.getTensor( offset + 2 ) )
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                typeImplementation.getKernelSource(), // kernelSource
                                "value = src1 + src2;\n",
                                "value += 1 * drain;\n",
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
        .setForwardADAnalyzer( call -> true )
        .setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
            {
                Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                Function mul = Function.Detached.MUL;
                if (
                    ctxDerivative != null
                ) {
                    return new DefaultADAgent( ctxDerivative )
                            .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                            .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{backwardError, ctxDerivative} ) );
                }
                Tsr[] inputs = call.getTensors();
                int d = call.getDerivativeIndex();
                if( forward )
                {
                    Tsr deriv = f.derive( inputs, d );
                    return new DefaultADAgent(
                            deriv
                        ).setForward(
                            ( t, derivative ) -> mul.call(new Tsr[]{derivative, deriv})
                        ).setBackward(
                            ( t, derivative ) -> mul.call( new Tsr[]{derivative, deriv} )
                        );
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

        setImplementation(Activation.class,
                activation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
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
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = ( call.getTensor( 0 ) != null ) ? 0 : 1;
                                    int gwz =
                                            ( call.getTensor( 0 ) != null )
                                                    ? call.getTensor( 0 ).size()
                                                    : call.getTensor( 1 ).size();
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
            double sum = 0;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                sum += src.get( 0 ).call( inputs, i );
                nothingDone = false;
            }
            if ( nothingDone ) {
                return src.get( 0 ).call( inputs );
            }
            return sum;
        } else {
            return src.get( 0 ).derive( inputs, d, j );
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, List<Function> src ) {
        if ( d < 0 ) {
            double sum = 0;
            boolean nothingDone = true;
            for (int i = 0; i < inputs.length; i++) {
                sum += src.get( 0 ).call( inputs, i );
                nothingDone = false;
            }
            if ( nothingDone ) {
                return src.get( 0 ).call( inputs );
            }
            return sum;
        } else {
            double sum = 0;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                double r = src.get( 0 ).derive( inputs, d, i );
                sum += r;
                nothingDone = false;
            }
            if ( nothingDone ) {
                return src.get( 0 ).call( inputs );
            }
            return sum;
        }

    }


}
