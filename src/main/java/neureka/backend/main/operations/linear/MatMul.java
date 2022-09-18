package neureka.backend.main.operations.linear;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.FunDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.implementations.matmul.CLMatMul;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
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
                .identifier(       "matMul"    )
                .operator(         "@"         )
                .arity(            2           )
                .isOperator(       true        )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );

        //throw new IllegalArgumentException("Matrix multiplication does not support call preparation!");
        // At least not through this way... this is the deprecated way...
        simpleMatMulAlgorithm =
            DeviceAlgorithm
                .withName("simple_matmul")
                .setIsSuitableFor(
                    call -> call.validate()
                                .allNotNull( t -> Number.class.isAssignableFrom(t.getItemType()) )
                                .getEstimator()
                                    .goodIfAnyNonNull( t -> t.getNDConf() instanceof Simple2DConfiguration)
                                    .badIfAnyNonNull( t -> !( t.getNDConf() instanceof Simple2DConfiguration) )
                                    .getEstimation()
                )
                .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
                .setDeviceExecution(
                    ( call, callback ) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm(call, null),
                    ( Function f, ExecutionCall<? extends Device<?>> adCall ) ->
                    {
                        if ( adCall.autogradMode().allowsForward() )
                            throw new IllegalArgumentException("Matrix multiplication does not support forward-AD!");
                        Function matMul = Neureka.get().backend().getFunction().matMul();
                        int d = ( 1 + adCall.getValOf( Arg.DerivIdx.class ) ) % 2;
                        Tsr<?> derivative = adCall.input( d ).T().deepCopy().getUnsafe().setIsIntermediate( true ); // We need to clone it to make it have a simple nd configuration...
                        derivative.to(adCall.getDevice());
                        return ADAction.of( target ->
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
                new CPUMatMul()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                new CLMatMul()
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
     *  This is usually the case when they are slices or reshaped views on data...
     *  The reason for this is simply that we need inline data for the OpenCL kernels!
     *
     *
     * @param call The execution call whose tensors ought to be cloned based on the complexity of their access patterns.
     */
    private static ExecutionCall<?> _autoClone( ExecutionCall<?> call ) {
        for (int i = 0; i < call.arity(); i++ ) {
            if (
                    (!_isSimpleRowMajorMatrix( call.input( i ) ) && !_isSimpleColumnMajorMatrix( call.input( i ) ))
                            ||
                    call.input( i ).isPartialSlice()
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
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
