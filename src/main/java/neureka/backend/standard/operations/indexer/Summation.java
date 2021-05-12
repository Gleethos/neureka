package neureka.backend.standard.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.standard.operations.JunctionUtil;
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
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Summation extends AbstractOperation
{

    public Summation()
    {
        super (
                new OperationBuilder()
                        .setFunction(         "sumJs"    )
                        .setOperator(         "sumJs"    )
                        .setArity(            1           )
                        .setIsOperator(       false       )
                        .setIsIndexer(        true        )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );

        Algorithm.RecursiveJunctor rja = JunctionUtil::forAdditions;

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

        DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX =
                ( inputs, d ) ->
                {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )] + t2_val[inputs[ 2 ].indexOfIndices(t2Idx)];
                    else return ( t0Idx, t1Idx, t2Idx ) -> 1.0;
                };

        Broadcast operationAlgorithm = new Broadcast()
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if ( ctxDerivative != null ) {
                            return new DefaultADAgent( ctxDerivative )
                                    .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                                    .setBackward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr deriv = f.derive( inputs, d );
                            return new DefaultADAgent( deriv )
                                    .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                    .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, deriv } ) );
                        }
                    }
                )
                .setHandleRecursivelyAccordingToArity( rja )
                .build();


        setAlgorithm(
                Broadcast.class,
                operationAlgorithm.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                                ? ( start, end ) ->
                                                                    Broadcast.broadcast (
                                                                            call.getTsrOfType( Number.class, 0 ),
                                                                            call.getTsrOfType( Number.class, 1 ),
                                                                            call.getTsrOfType( Number.class, 2 ),
                                                                            call.getDerivativeIndex(),
                                                                            start, end,
                                                                            _creatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                    )
                                                                :  ( start, end ) ->
                                                                    Broadcast.broadcast (
                                                                            call.getTsrOfType( Number.class, 0 ),
                                                                            call.getTsrOfType( Number.class, 1 ),
                                                                            call.getTsrOfType( Number.class, 2 ),
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
                                    int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTsrOfType( Number.class, offset ) )
                                            .pass( call.getTsrOfType( Number.class, offset + 1 ) )
                                            .pass( call.getTsrOfType( Number.class, offset + 2 ) )
                                            .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                operationAlgorithm.getKernelSource(), // kernelSource
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

        DefaultOperatorCreator<TertiaryNDAConsumer> activationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                    else return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                };

        Activation activation = new Activation()
        .setCanPerformBackwardADFor( call -> true )
        .setCanPerformForwardADFor( call -> true )
        .setSupplyADAgentFor(
            ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
            {
                Tsr ctxDerivative = (Tsr) call.getAt("derivative");
                Function mul = Function.Detached.MUL;
                if ( ctxDerivative != null )
                    return new DefaultADAgent( ctxDerivative )
                            .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                            .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, ctxDerivative } ) );

                Tsr[] inputs = call.getTensors();
                int d = call.getDerivativeIndex();
                if ( forward )
                {
                    Tsr deriv = f.derive( inputs, d );
                    return new DefaultADAgent( deriv )
                            .setForward( ( t, derivative ) -> mul.call( derivative, deriv ) )
                            .setBackward( ( t, derivative ) -> mul.call( new Tsr[]{derivative, deriv} ) );
                }
                else
                {
                    if ( this.supports(Convolution.class) )
                    {
                        Function invX = new FunctionBuilder(OperationContext.get()).build(
                                "I[ 0 ]" + getOperator() + ">>I[ 1 ]" + getOperator() + ">>I[ 2 ]",
                                false
                        );
                        Tsr deriv = f.derive( inputs, d );
                        return new DefaultADAgent( deriv )
                                .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                .setBackward( (t, error) -> invX.call( error, deriv, new Tsr(t.getPayload().shape(), 0) ) );
                    }
                    else
                    {
                        Tsr deriv = f.derive( inputs, d );
                        return new DefaultADAgent( deriv )
                                .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, deriv } ) );
                    }
                }
            }
        )
        .setHandleInsteadOfDevice( (caller, call ) -> null )
        .setHandleRecursivelyAccordingToArity( rja )
        .setInstantiateNewTensorsForExecutionIn(
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

        setAlgorithm(
                Activation.class,
                activation.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        start, end,
                                                                        activationXCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ),
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
                                    int offset = ( call.getTsrOfType( Number.class, 0 ) != null ) ? 0 : 1;
                                    int gwz =
                                            ( call.getTsrOfType( Number.class, 0 ) != null )
                                                    ? call.getTsrOfType( Number.class, 0 ).size()
                                                    : call.getTsrOfType( Number.class, 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTsrOfType( Number.class, offset ) )
                                            .pass( call.getTsrOfType( Number.class, offset + 1 ) )
                                            .pass( call.getTsrOfType( Number.class, 0 ).rank() )
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
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if (expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')') {
            return "sumJs" + expression;
        }
        return "sumJs" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int d ) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double sum = 0;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                sum += src[ 0 ].call( inputs, i );
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs );
            return sum;
        }
        else return src[ 0 ].derive( inputs, d, j );
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double sum = 0;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                sum += src[ 0 ].call( inputs, i );
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs );
            return sum;
        } else {
            double sum = 0;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                double r = src[ 0 ].derive( inputs, d, i );
                sum += r;
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs );
            return sum;
        }

    }


}
