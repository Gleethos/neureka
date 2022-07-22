package neureka.backend.main.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.operations.JunctionUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

/**
 *  This type of operation belongs to the same species as the
 *  {@link Summation} operation.
 *  It executes incoming calls so that the calling function
 *  will be executed with all input indices passed to it.
 *  The resulting array of tensors will then multiplied with each other
 *  to produce the result of this operation, hence the name {@link Product}.
 */
public final class Product extends AbstractOperation
{
    public Product()
    {
        super (
            new OperationBuilder()
                    .setIdentifier(         "prodJs"    )
                    .setOperator(         "prodJs"    )
                    .setArity(            1           )
                    .setIsOperator(       false       )
                    .setIsIndexer(        true        )
                    .setIsDifferentiable( true        )
                    .setIsInline(         false       )
        );

        //________________
        // BROADCASTING :

        setAlgorithm(
            new Broadcast(JunctionUtil::forMultiplications)
            .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
            .setSupplyADAgentFor(
                ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                {
                    if ( call.autogradMode().allowsForward() )
                        throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                    Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                    Function mul = Neureka.get().backend().getFunction().mul();
                    if ( ctxDerivative != null ) {
                        return ADAgent.of( ctxDerivative )
                                        .withAD( target -> mul.execute( target.error(), ctxDerivative ) );
                    }
                    int d = call.getValOf( Arg.DerivIdx.class );
                    Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                    return ADAgent.of( derivative )
                                    .withAD( target -> mul.execute( target.error(), derivative ) );
                }
            )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                Broadcast.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> a * b,
                        ( a, b ) -> b, // Deriving at input 0
                        ( a, b ) -> a  // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> a * b,
                        ( a, b ) -> b, // Deriving at input 0
                        ( a, b ) -> a  // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                Broadcast.implementationForGPU( this.getIdentifier() )
                    .with( "value = src1 * src2;\n" )
                    .and( "value += handle * drain;\n" )
            )
        );

        //______________
        // ACTIVATION :

        Activation activation = new Activation()
        .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
        .setDeviceExecution(
            JunctionUtil::forMultiplications,
            (Function f, ExecutionCall<? extends Device<?>> adCall ) -> // Autograd
            {
                Function mul = Neureka.get().backend().getFunction().mul();
                Tsr<?> derivative = f.executeDerive( adCall.inputs(), adCall.getDerivativeIndex() );
                return ADAgent.of( derivative )
                                .withAD( target -> mul.execute( target.error(), derivative ) );
            }
        )
        .setCallPreparation(
            call -> {
                Device<Number> device = call.getDeviceFor(Number.class);
                if ( call.input( 0 ) == null ) // Creating a new tensor:
                {
                    int[] shp = call.input( 1 ).getNDConf().shape();
                    Tsr<Double> output = Tsr.of( shp, 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    device.store( output );
                    call.setInput( 0, output );
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
                        Activation.implementationForCPU()
                            .with(Fun.F64ToF64.pair(
                                x -> x,
                                x -> x
                            ))
                            .with(Fun.F32ToF32.pair(
                                x -> x,
                                x -> x
                            ))
                            .with(Fun.I32ToI32.pair(
                                x -> x,
                                x -> x
                            ))
                            .get()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    CLImplementation.compiler()
                            .arity( 3 )
                            .kernelSource( activation.getKernelSource() )
                            .activationSource( "output = input;" )
                            .differentiationSource( "output = 1;" )
                            .kernelPostfix( this.getIdentifier() )
                            .execution(
                                call -> {
                                    int offset = (call.input( Number.class, 0 ) != null) ? 0 : 1;
                                    int gwz = (call.input( Number.class, 0 ) != null) ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .passAllOf( call.input( Number.class, offset ) )
                                            .passAllOf( call.input( Number.class, offset + 1 ) )
                                            .pass( call.input( Number.class, 0 ).rank() )
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
