package neureka.backend.standard.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.CalcUtil;
import neureka.calculus.args.Arg;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
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

        Broadcast operationAlgorithm = new Broadcast(JunctionUtil::forAdditions)
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        Function mul = Neureka.get().context().getFunction().mul();
                        if ( ctxDerivative != null ) {
                            return DefaultADAgent.ofDerivative( ctxDerivative )
                                    .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                                    .setBackward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getValOf( Arg.DerivIdx.class );
                        if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr deriv = f.derive( inputs, d );
                            return DefaultADAgent.ofDerivative( deriv )
                                    .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                    .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, deriv } ) );
                        }
                    }
                )
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
                                                        (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                                ? ( start, end ) ->
                                                                    Broadcast.broadcast (
                                                                            call.getTsrOfType( Number.class, 0 ),
                                                                            call.getTsrOfType( Number.class, 1 ),
                                                                            call.getTsrOfType( Number.class, 2 ),
                                                                            call.getValOf( Arg.DerivIdx.class ),
                                                                            start, end,
                                                                            _creatorX.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                    )
                                                                :  ( start, end ) ->
                                                                    Broadcast.broadcast (
                                                                            call.getTsrOfType( Number.class, 0 ),
                                                                            call.getTsrOfType( Number.class, 1 ),
                                                                            call.getTsrOfType( Number.class, 2 ),
                                                                            call.getValOf( Arg.DerivIdx.class ),
                                                                            start, end,
                                                                            _creator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                    )
                                                ),
                                3
                        )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operationAlgorithm.getKernelSource() )
                                .activationSource( "value = src1 + src2;\n" )
                                .differentiationSource( "value += 1 * drain;\n" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                            call.getDevice().getKernel(call)
                                                    .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                                    .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                                    .passAllOf( call.getTsrOfType( Number.class, offset + 2 ) )
                                                    .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                                    .call( gwz );
                                        }
                                )
                                .build()
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
                Tsr ctxDerivative = (Tsr) call.getValOf(Arg.Derivative.class);
                Function mul = Neureka.get().context().getFunction().mul();
                if ( ctxDerivative != null )
                    return DefaultADAgent.ofDerivative( ctxDerivative )
                            .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                            .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, ctxDerivative } ) );

                Tsr[] inputs = call.getTensors();
                int d = call.getDerivativeIndex();
                if ( forward )
                {
                    Tsr deriv = f.derive( inputs, d );
                    return DefaultADAgent.ofDerivative( deriv )
                            .setForward( ( t, derivative ) -> mul.call( derivative, deriv ) )
                            .setBackward( ( t, derivative ) -> mul.call( new Tsr[]{derivative, deriv} ) );
                }
                else
                {
                    if ( this.supports(Convolution.class) )
                    {
                        Function invX = new FunctionBuilder( Neureka.get().context() ).build(
                                "I[ 0 ]" + getOperator() + ">>I[ 1 ]" + getOperator() + ">>I[ 2 ]",
                                false
                        );
                        Tsr deriv = f.derive( inputs, d );
                        return DefaultADAgent.ofDerivative( deriv )
                                .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                .setBackward( (t, error) -> invX.execute( error, deriv, Tsr.of(t.getPayload().shape(), 0) ) );
                    }
                    else
                    {
                        Tsr deriv = f.derive( inputs, d );
                        return DefaultADAgent.ofDerivative( deriv )
                                .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, deriv } ) );
                    }
                }
            }
        )
        .setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, JunctionUtil::forAdditions ) )
        .setCallPreparation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[ 1 ].getNDConf().shape();
                        Tsr output = Tsr.of( shp, 0.0 );
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
                                                        (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        start, end,
                                                                        activationXCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ),
                                                                        start, end,
                                                                        activationCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                ),
                                3
                        )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( activation.getKernelSource() )
                                .activationSource( "output = input;" )
                                .differentiationSource( "output = 1;" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = ( call.getTsrOfType( Number.class, 0 ) != null ) ? 0 : 1;
                                            int gwz =
                                                    ( call.getTsrOfType( Number.class, 0 ) != null )
                                                            ? call.getTsrOfType( Number.class, 0 ).size()
                                                            : call.getTsrOfType( Number.class, 1 ).size();
                                            call.getDevice().getKernel(call)
                                                    .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                                    .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                                    .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                                    .call( gwz );
                                        }
                                )
                                .build()
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
    public String asDerivative( Function[] children, int derivationIndex) {
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
