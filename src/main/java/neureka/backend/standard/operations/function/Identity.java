package neureka.backend.standard.operations.function;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.Fun;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Identity extends AbstractOperation
{
    public Identity()
    {
        super(
            new OperationBuilder()
                    .setFunction(         "idy"    )
                    .setOperator(         "idy"    )
                    .setArity(            1        )
                    .setIsOperator(       false    )
                    .setIsIndexer(        false    )
                    .setIsDifferentiable( true     )
                    .setIsInline(         false    )
        );

        Activation operationAlgorithm = new Activation()
        .setCanPerformBackwardADFor( call -> true )
        .setCanPerformForwardADFor(
            call -> {
                Tsr<?> last = null;
                for ( Tsr<?> t : call.getTensors() ) {
                    if ( last != null && !last.shape().equals(t.shape()) ) return false;
                    last = t; // Note: shapes are cached!
                }
                return true;
            }
        )
        .setSupplyADAgentFor( getDefaultAlgorithm() )
        .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
        .setCallPreparation(
            call -> {
                Tsr<?>[] tensors = call.getTensors();
                int offset = ( tensors[ 0 ] == null ) ? 1 : 0;
                return ExecutionCall.of(tensors[offset], tensors[1+offset])
                                    .andArgs(Arg.DerivIdx.of(-1))
                                    .running(Neureka.get().backend().getOperation("idy"))
                                    .on(call.getDevice());
            }
        )
        .buildFunAlgorithm();

        setAlgorithm(
            Activation.class,
            operationAlgorithm.setImplementationFor(
                CPU.class,
                CPUImplementation
                    .withArity(2)
                    .andImplementation(
                        Activation.implementationForCPU()
                            .with(Fun.F64ToF64.pair(
                                    x -> x,
                                    x -> 1
                            ) )
                            .with(Fun.F32ToF32.pair(
                                    x -> x,
                                    x -> 1
                            ))
                            .get()
                    )
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation.compiler()
                    .arity( 2 )
                    .kernelSource( operationAlgorithm.getKernelSource() )
                    .activationSource( "output = input;\n" )
                    .differentiationSource( "output = input;\n" )
                    .kernelPostfix( this.getFunction() )
                    .execution(
                        call -> {
                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                            // Drain tensor needs to be 'actual'! :
                            if (!call.getTsrOfType( Number.class, offset + 1).isVirtual()) call.getTsrOfType( Number.class, offset).setIsVirtual( false );
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

        Scalarization scalarization = new Scalarization()
            .setCanPerformBackwardADFor( call -> true )
            .setCanPerformForwardADFor(
                    call -> {
                        Tsr<?> last = null;
                        for ( Tsr<?> t : call.getTensors() ) {
                            if ( last != null && !last.shape().equals(t.shape()) ) return false;
                            last = t; // Note: shapes are cached!
                        }
                        return true;
                    }
            )
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
            .setCallPreparation(
                call -> {
                    Tsr<?>[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[ 1 ].getNDConf().shape();
                        Tsr<?> output = Tsr.of( shp, 0.0 ).getUnsafe().setIsIntermediate( true );
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
            Scalarization.class,
            scalarization.setImplementationFor(
                CPU.class,
                CPUImplementation
                    .withArity(2)
                    .andImplementation(
                        Scalarization.implementationForCPU()
                            .with(Fun.F64F64ToF64.triple(
                                ( a, b ) -> b,
                                ( a, b ) -> b, // Deriving at input 0
                                ( a, b ) -> b // deriving input 1
                            ))
                            .with(Fun.F32F32ToF32.triple(
                                    ( a, b ) -> b,
                                    ( a, b ) -> b, // Deriving at input 0
                                    ( a, b ) -> b // deriving input 1
                            ))
                            .get()
                    )
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation.compiler()
                    .arity( 2 )
                    .kernelSource( scalarization.getKernelSource() )
                    .activationSource( "output = value;\n" )
                    .differentiationSource( "output = value;\n" )
                    .kernelPostfix( this.getFunction() )
                    .execution(
                        call -> {
                            Tsr<Number> t = call.getTsrOfType( Number.class, 0 );
                            int gwz = t.size();
                            call.getDevice()
                                .getKernel(call)
                                .passAllOf(t)
                                .passAllOf(t)
                                .pass((float)call.getTsrOfType( Number.class, 1 ).getDataAs( double[].class )[ 0 ])
                                .pass(t.rank())
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
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "idy" + expression;
        return "idy" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return calculate(
                src[ 0 ].call( inputs, j ),
                d >= 0
            ) * ( ( d < 0 ) ? 1 : src[ 0 ].derive( inputs, d, j ) );
    }

    @Contract(pure = true)
    public static double calculate(double input, boolean derive) {
        if ( !derive ) return input;
        else return 1;
    }



}
