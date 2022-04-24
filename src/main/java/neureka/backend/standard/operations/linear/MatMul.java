package neureka.backend.standard.operations.linear;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.fun.ADAgentSupplier;
import neureka.backend.api.algorithms.fun.AutoDiff;
import neureka.backend.api.algorithms.fun.Result;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.FunAlgorithm;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.backend.standard.operations.linear.internal.opencl.GEMM;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.simple.Simple2DConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MatMul extends AbstractOperation
{
    private static final Logger _LOG = LoggerFactory.getLogger(MatMul.class);
    private final FunAlgorithm simpleMatMulAlgorithm;

    public MatMul()
    {
        super(
                new OperationBuilder()
                        .setIdentifier(         "matMul"    )
                        .setOperator(         "@"         )
                        .setArity(            2           )
                        .setIsOperator(       true        )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );

        simpleMatMulAlgorithm =
                        Algorithm
                            .withName("simple_matmul")
                            .setIsSuitableFor(
                                call -> call.validate()
                                            .allNotNull( t -> Number.class.isAssignableFrom(t.getValueClass()) )
                                            .getEstimator()
                                                .goodIfAnyNonNull( t -> t.getNDConf() instanceof Simple2DConfiguration)
                                                .badIfAnyNonNull( t -> !( t.getNDConf() instanceof Simple2DConfiguration) )
                                                .getEstimation()
                            )
                            .setAutogradModeFor( call -> AutoDiff.BACKWARD_ONLY )
                            .setExecution(
                                ( caller, call ) -> {
                                    ADAgentSupplier autoDiff = ( Function f, ExecutionCall<? extends Device<?>> adCall, boolean forward ) ->
                                    {
                                        if ( forward ) throw new IllegalArgumentException("Matrix multiplication does not support forward-AD!");

                                        Function matMul = Neureka.get().backend().getFunction().matMul();
                                        int d = ( 1 + adCall.getValOf( Arg.DerivIdx.class ) ) % 2;
                                        Tsr<?> derivative = adCall.input( d ).T().clone().getUnsafe().setIsIntermediate( true ); // We need to clone it to make it have a simple nd configuration...
                                        derivative.to(adCall.getDevice());
                                        return ADAgent.of( derivative )
                                                .setBackward( (node, error) -> {
                                                    if ( d == 1 )
                                                        return matMul.execute( error, derivative );
                                                    else
                                                        return matMul.execute( derivative, error );
                                                });
                                    };

                                    if ( !caller.isFlat() )
                                        return Result.of(CalcUtil.defaultRecursiveExecution( caller, call )).withADAgent(autoDiff);

                                    Tsr<?>[] tensors = CalcUtil.srcActivation(call.inputs(), call.getValOf( Arg.VarIdx.class ), -1, 1, caller.getSubFunctions().toArray(new Function[0]));
                                    for ( Tsr<?> t : tensors ) if ( t != null ) t.setIsVirtual( false );
                                    ExecutionCall<Device<Object>> preparedCall = _prepare( call.withInputs(tensors) );
                                    return Result.of(MatMul.this.simpleMatMulAlgorithm
                                                        .getImplementationFor(call.getDeviceFor(Object.class))
                                                        .runAndGetFirstTensor(preparedCall)).withADAgent(autoDiff);
                                }
                            )
                            .setCallPreparation( MatMul::_prepare )
                            .buildFunAlgorithm();

        setAlgorithm(
            simpleMatMulAlgorithm
                .setImplementationFor(
                    CPU.class,
                    CPUImplementation
                        .withArity(3)
                        .andImplementation( new CPUMatMul() )
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    CLImplementation
                        .fromSource()
                        .arity( 3 )
                        .kernelName( "simple_matMul" )
                        .kernelSource(
                            "   __kernel void simple_matMul(                                         \n" +
                            "          const int M, const int N, const int K,                        \n" +
                            "          const __global float* A,                                      \n" +
                            "          const __global float* B,                                      \n" +
                            "                __global float* C                                       \n" +
                            "   ) {                                                                  \n" +
                            "       const int m = get_global_id(0); // Row index of C (0..M)         \n" +
                            "       const int n = get_global_id(1); // Col index of C (0..N)         \n" +
                            "                                                                        \n" +
                            "       // Compute a single element (loop over K)                        \n" +
                            "       float acc = 0.0f;                                                \n" +
                            "       for ( int k = 0; k < K; k++ )                                    \n" +
                            "           acc += A[ k + m * K ] * B[ n + k * N ];                      \n" +
                            "                                                                        \n" +
                            "       // Store the result                                              \n" +
                            "       C[ n + m * N ] = acc;                                            \n" +
                            "   }                                                                    \n"
                        )
                        .lambda( call -> {
                            if (
                                call.validate()
                                        .all( t -> t.getNDConf().getLayout() == NDConfiguration.Layout.COLUMN_MAJOR )
                                        .isValid()
                            ) {
                                new GEMM().run( call );
                            } else {
                                int M = call.input(1).shape(0);
                                int N = call.input(2).shape(1);
                                int K = call.input(1).shape(1);
                                call.getDevice()
                                        .getKernel(call)
                                        .pass(M).pass(N).pass(K)
                                        .pass(call.input(Number.class, 1))
                                        .pass(call.input(Number.class, 2))
                                        .pass(call.input(Number.class, 0))
                                        .call(new long[]{M, N}, null);
                            }
                        })
                        .build()
                )
        );


    }

    private static ExecutionCall<Device<Object>> _prepare( ExecutionCall call )
    {
        Device<Number> device = call.getDeviceFor(Number.class);
        if ( call.input( 0 ) == null ) // Creating a new tensor:
        {
            Class<Number> type = (Class<Number>) call.input(  1 ).getDataType().getJVMTypeClass();
            int[] shp = new int[]{ call.input( 1 ).shape(0), call.input( 2 ).shape(1) };
            NDConfiguration.Layout targetLayout = call.input( 1 ).getNDConf().getLayout();
            call.input( 2 ).getUnsafe().toLayout(targetLayout);
            Tsr<Number> output = Tsr.of( type ).withShape( shp ).all( 0 ).getUnsafe().setIsIntermediate( true );
            output.getUnsafe().toLayout(targetLayout);
            output.setIsVirtual( false ); // This statement is after the layout conversion for performance reasons (virtual tensors barely need copying).
            try {
                device.store( output );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
            call.setInput( 0, output );
        }
        _autoClone( call );
        return (ExecutionCall<Device<Object>>) call;
    }

    /**
     *  This method will clone {@link Tsr} instances if they do not
     *  possess a simple {@link neureka.ndim.config.NDConfiguration}.
     *
     * @param call The execution call whose tensors ought to be cloned based on the complexity of their access patterns.
     */
    private static void _autoClone( ExecutionCall<?> call ) {
        for (int i = 0; i < call.arity(); i++ ) {
            if (
                    !_isSimpleRowMajorMatrix( call.input( i ) )
                            &&
                    !_isSimpleColumnMajorMatrix( call.input( i ) )
            ) {
                _LOG.debug("Auto cloning a tensor which does not have a simple ND configuration...");
                call.setInput( i, call.input( i ).clone().getUnsafe().setIsIntermediate( true ) );
                /*
                    The user should do cloning explicitly because using slices
                    will cause the backend to perform auto cloning every time the
                    slice is being used for operations like this one...
                 */
            }
        }
    }

    private static boolean _isSimpleColumnMajorMatrix( Tsr<?> t ) {
        return t.rank() == 2 && t.getNDConf().getLayout() == NDConfiguration.Layout.COLUMN_MAJOR;
    }

    private static boolean _isSimpleRowMajorMatrix( Tsr<?> t ) {
        return t.rank() == 2 && t.getNDConf().getLayout() == NDConfiguration.Layout.ROW_MAJOR;
    }

    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 )
                reconstructed.append(" @ ");
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }
}
