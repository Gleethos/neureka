package neureka.backend.main.operations.linear;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.api.template.algorithms.FunDeviceAlgorithm;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.implementations.CPUImplementation;
import neureka.backend.main.operations.linear.internal.opencl.GEMM;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.backend.main.internal.AlgoUtil;
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
    private final FunDeviceAlgorithm simpleMatMulAlgorithm;


    public MatMul()
    {
        super(
            new OperationBuilder()
                .setIdentifier(       "matMul"    )
                .setOperator(         "@"         )
                .setArity(            2           )
                .setIsOperator(       true        )
                .setIsIndexer(        false       )
                .setIsDifferentiable( true        )
                .setIsInline(         false       )
        );

        //throw new IllegalArgumentException("Matrix multiplication does not support call preparation!");
        // At least not through this way... this is the deprecated way...
        simpleMatMulAlgorithm =
            DeviceAlgorithm
                .withName("simple_matmul")
                .setIsSuitableFor(
                    call -> call.validate()
                                .allNotNull( t -> Number.class.isAssignableFrom(t.getItemClass()) )
                                .getEstimator()
                                    .goodIfAnyNonNull( t -> t.getNDConf() instanceof Simple2DConfiguration)
                                    .badIfAnyNonNull( t -> !( t.getNDConf() instanceof Simple2DConfiguration) )
                                    .getEstimation()
                )
                .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
                .setDeviceExecution(
                        ( context, callback ) -> AlgoUtil.executeDeviceAlgorithm(context.call(), null),
                        ( Function f, ExecutionCall<? extends Device<?>> adCall ) ->
                        {
                            if ( adCall.autogradMode().allowsForward() )
                                throw new IllegalArgumentException("Matrix multiplication does not support forward-AD!");
                            Function matMul = Neureka.get().backend().getFunction().matMul();
                            int d = ( 1 + adCall.getValOf( Arg.DerivIdx.class ) ) % 2;
                            Tsr<?> derivative = adCall.input( d ).T().deepCopy().getUnsafe().setIsIntermediate( true ); // We need to clone it to make it have a simple nd configuration...
                            derivative.to(adCall.getDevice());
                            return ADAgent.of( derivative )
                                    .withAD( target ->
                                            d == 1
                                                    ? matMul.execute( target.error(), derivative )
                                                    : matMul.execute( derivative, target.error() )
                                    );
                    }
                )
                .setCallPreparation(MatMul::_prepare)
                .buildFunAlgorithm();

        setAlgorithm(
            simpleMatMulAlgorithm
            .setImplementationFor(
                CPU.class,
                CPUImplementation
                .withArity(3)
                .andFunImplementation( new CPUMatMul() )
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
        assert call.arity() <= 3;
        Device<Number> device = call.getDeviceFor(Number.class);
        if ( call.arity() == 2 ) call = call.withAddedInputAt(0, null);
        if ( call.input( 0 ) == null ) // Creating a new tensor:
        {
            Class<Number> type = (Class<Number>) call.input(  1 ).getDataType().getItemTypeClass();
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
            call = call.withInputAt( 0, output );
        }
        return (ExecutionCall<Device<Object>>) _autoClone( call );
    }

    /**
     *  This method will clone {@link Tsr} instances if they do not
     *  possess a simple {@link neureka.ndim.config.NDConfiguration}.
     *
     * @param call The execution call whose tensors ought to be cloned based on the complexity of their access patterns.
     */
    private static ExecutionCall<?> _autoClone( ExecutionCall<?> call ) {
        for (int i = 0; i < call.arity(); i++ ) {
            if (
                    !_isSimpleRowMajorMatrix( call.input( i ) )
                            &&
                    !_isSimpleColumnMajorMatrix( call.input( i ) )
            ) {
                _LOG.debug("Auto cloning a tensor which does not have a simple ND configuration...");
                call = call.withInputAt( i, call.input( i ).deepCopy().getUnsafe().setIsIntermediate( true ) );
                /*
                    The user should do cloning explicitly because using slices
                    will cause the backend to perform auto cloning every time the
                    slice is being used for operations like this one...
                 */
            }
        }
        return call;
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
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
