package neureka.backend.standard.operations.function;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.Fun;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Sinus extends AbstractOperation
{
    public Sinus()
    {
        super(
            new OperationBuilder()
                .setFunction(         "sin"    )
                .setOperator(         "sin"    )
                .setArity(            1        )
                .setIsOperator(       false    )
                .setIsIndexer(        false    )
                .setIsDifferentiable( true     )
                .setIsInline(         false    )
        );

        Activation operationAlgorithm =
                new Activation()
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
                    .setSupplyADAgentFor(
                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                    )
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
                                 } catch ( Exception e ) {
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
            operationAlgorithm.setImplementationFor(
                CPU.class,
                Activation.implementationForCPU()
                    .with(Fun.F64ToF64.pair(
                        x -> Math.sin(x),
                        x -> Math.cos(x)
                    ))
                    .with(Fun.F32ToF32.pair(
                        x -> (float) Math.sin(x),
                        x -> (float) Math.cos(x)
                    ))
                    .with(Fun.I32ToI32.pair(
                        x -> (int) Math.round(Math.sin(x)),
                        x -> (int) Math.round(Math.cos(x))
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                    Activation.implementationForGPU( this.getFunction() )
                            .with( "output = sin( input );\n" )
                            .and( "output = cos( input );\n" )
            )
        );
    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "sin" + expression;
        return "sin" + "(" + expression + ")";
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
    public static double calculate(double input, boolean derive ) {
        if ( !derive ) return Math.sin( input );
        else return Math.cos( input );
    }




}
