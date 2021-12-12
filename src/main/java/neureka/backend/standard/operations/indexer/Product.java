package neureka.backend.standard.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Product extends AbstractOperation {


    public Product()
    {
        super (
                new OperationBuilder()
                        .setFunction(         "prodJs"    )
                        .setOperator(         "prodJs"    )
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
                    double[] t1_val = inputs[ 1 ].getDataAs( double[].class );
                    double[] t2_val = inputs[ 2 ].getDataAs( double[].class );
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] * t2_val[t2Idx.i()];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if ( d == 0 ) return t2_val[t2Idx.i()];
                            else return t1_val[ t1Idx.i() ];
                        };
                    }
                };

        DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX =
                ( inputs, d ) ->
                {
                    double[] t1_val = inputs[ 1 ].getDataAs( double[].class );
                    double[] t2_val = inputs[ 2 ].getDataAs( double[].class );
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )] * t2_val[inputs[ 2 ].indexOfIndices(t2Idx)];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if ( d == 0 ) return t2_val[inputs[ 2 ].indexOfIndices(t2Idx)];
                            else return t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                        };
                    }
                };

        Broadcast operationAlgorithm = new Broadcast(JunctionUtil::forMultiplications)
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        Function mul = Neureka.get().backend().getFunction().mul();
                        if ( ctxDerivative != null ) {
                                return ADAgent.of( ctxDerivative )
                                                .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) )
                                                .setBackward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) );
                        }
                        Tsr<?>[] inputs = call.getTensors();
                        int d = call.getValOf( Arg.DerivIdx.class );
                        if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr<?> derivative = f.executeDerive( inputs, d );
                            return ADAgent.of( derivative )
                                    .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                    .setBackward( (node, backwardError ) -> mul.execute( backwardError, derivative ) );
                        }
                    }
                )
                .buildFunAlgorithm();

        setAlgorithm(
                Broadcast.class,
                operationAlgorithm.setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
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
                                                )
                            )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operationAlgorithm.getKernelSource() )
                                .activationSource( "value = src1 * src2;\n" )
                                .differentiationSource( "value += handle * drain;\n" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = ( call.getTsrOfType( Number.class, 0 ) != null ) ? 0 : 1;
                                            int gwz = ( call.getTsrOfType( Number.class, 0 ) != null ) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
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
                    double[] t1_val = inputs[ 1 ].getDataAs( double[].class );
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                    else return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                };

        DefaultOperatorCreator<TertiaryNDAConsumer> activationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].getDataAs( double[].class );
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                    else return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                };

        Activation activation = new Activation()
        .setCanPerformBackwardADFor( call -> true )
        .setCanPerformForwardADFor( call -> true )
        .setSupplyADAgentFor(
            ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        Function mul = Neureka.get().backend().getFunction().mul();
                        if ( ctxDerivative != null ) {
                            return ADAgent.of( ctxDerivative )
                                .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) )
                                .setBackward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) );
                        }
                        Tsr<?>[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if ( forward )
                        {
                            Tsr<?> derivative = f.executeDerive( inputs, d );
                            return ADAgent.of( derivative )
                                    .setForward( (t, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                    .setBackward( (t, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) );
                        }
                        else
                        {
                            if ( this.supports(Convolution.class) )
                            {
                                Function deConv = new FunctionBuilder( Neureka.get().backend() ).build(
                                                            "I[ 0 ]" + getOperator() + ">>I[ 1 ]" + getOperator() + ">>I[ 2 ]",
                                                            false
                                                        );
                                Tsr<?> derivative = f.executeDerive( inputs, d );
                                return ADAgent.of( derivative )
                                        .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                        .setBackward( (t, error) -> deConv.execute( error, derivative, Tsr.of(t.getPayload().shape(), 0) ) );
                            }
                            else
                            {
                                Tsr<?> derivative = f.executeDerive( inputs, d );
                                return ADAgent.of( derivative )
                                        .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                        .setBackward( (node, backwardError ) -> mul.execute( backwardError, derivative ) );
                            }
                        }
                    }
        )
        .setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, JunctionUtil::forMultiplications ) )
        .setCallPreparation(
                call -> {
                    Tsr<?>[] tsrs = call.getTensors();
                    Device<Number> device = call.getDeviceFor(Number.class);
                    if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[ 1 ].getNDConf().shape();
                        Tsr<Double> output = Tsr.of( shp, 0.0 );
                        output.setIsVirtual( false );
                        try {
                            device.store( output );
                        } catch( Exception e ) {
                            e.printStackTrace();
                        }
                        tsrs[ 0 ] = output;
                    }
                    return call;
                }
        )
        .buildFunAlgorithm();

        setAlgorithm(
                Activation.class,
                activation
                    .setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
                                call  ->
                                        call.getDevice()
                                                .getExecutor()
                                                .threaded(
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
                                                )
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
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
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
        if ( expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')' ) {
            return "prodJs" + expression;
        }
        return "prodJs" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src )
    {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double prod = 1;
            boolean nothingDone = true;
            for ( int Ii = 0; Ii < inputs.length; Ii++ ) {
                prod *= src[ 0 ].call( inputs, Ii );
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs, j );
            return prod;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs, 0 );
            ud = src[ 0 ].derive(inputs, d, 0);
            for ( int ji = 1; ji < inputs.length; ji++ ) {
                v = src[ 0 ].call( inputs, ji );
                vd = src[ 0 ].derive( inputs, d, ji );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double prod = 1;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                prod *= src[ 0 ].call( inputs, i );
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs );
            return prod;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call(inputs, 0);
            ud = src[ 0 ].derive(inputs, d, 0);
            for ( int j = 1; j < inputs.length; j++ ) {
                v = src[ 0 ].call( inputs, j );
                vd = src[ 0 ].derive( inputs, d, j );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }


}
