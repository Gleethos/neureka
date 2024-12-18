package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tensor;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.devices.Device;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.simple.Simple2DConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MatMulAlgorithm extends AbstractFunDeviceAlgorithm<MatMulAlgorithm>
{
    private static final Logger _LOG = LoggerFactory.getLogger(MatMulAlgorithm.class);

    public MatMulAlgorithm() {
        super("simple_matmul");
        setIsSuitableFor(
                call -> call.validate()
                        .allNotNull( t -> Number.class.isAssignableFrom(t.getItemType()) )
                        .getEstimator()
                        .goodIfAnyNonNull( t -> t.getNDConf() instanceof Simple2DConfiguration)
                        .badIfAnyNonNull( t -> !( t.getNDConf() instanceof Simple2DConfiguration) )
                        .getEstimation()
        );
        setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY );
        setExecution(
            (outerCaller, outerCall) ->
                Result.of(AbstractDeviceAlgorithm.executeDeviceAlgorithm(_prepare(outerCall)))
                .withAutoDiff( (Function f, ExecutionCall<? extends Device<?>> adCall ) ->
                {
                    if ( adCall.autogradMode().allowsForward() )
                        throw new IllegalArgumentException("Matrix multiplication does not support forward-AD!");
                    Function matMul = Neureka.get().backend().getFunction().matMul();
                    int d = ( 1 + adCall.getValOf( Arg.DerivIdx.class ) ) % 2;
                    Tensor<?> derivative = Util.transpose(adCall.input( d )).deepCopy().mut().setIsIntermediate( true ); // We need to clone it to make it have a simple nd configuration...
                    derivative.to(adCall.getDevice());
                    return ADAction.of(target -> {
                        Tensor<?> result;
                        switch ( d ) {
                            case 0:
                                result = matMul.execute(derivative, target.error());
                                break;
                            case 1:
                                result = matMul.execute(target.error(), derivative);
                                break;
                            default:
                                throw new IllegalStateException("This should never happen!");
                        }
                        return result;
                    });
                })
        );
        setCallPreparation(MatMulAlgorithm::_prepare);
    }

    private static ExecutionCall<Device<Object>> _prepare( ExecutionCall<?> call )
    {
        assert call.arity() <= 3;
        if ( call.arity() == 2 ) call = call.withAddedInputAt(0, null);
        if ( call.input( 0 ) == null ) // Creating a new tensor:
            call = _withNewOutput( call );

        return (ExecutionCall<Device<Object>>) _autoClone( call );
    }

    private static ExecutionCall<?> _withNewOutput( ExecutionCall<?> call )
    {
        Class<Number> type = (Class<Number>) call.input(  1 ).getDataType().getItemTypeClass();

        int[] shp = new int[]{ call.input( 1 ).shape(0), call.input( 2 ).shape(1) };
        Tensor<Number> output = Tensor.of( type ).withShape( shp ).all( 0 ).mut().setIsIntermediate( true );

        call = _checkAndPrepareLayout( call, output );

        call.getDeviceFor(Number.class).store( output );
        return call.withInputAt( 0, output );
    }

    private static ExecutionCall<?> _checkAndPrepareLayout( ExecutionCall<?> call, Tensor<?> c )
    {
        Tensor<?> a = call.input( 1 );
        Tensor<?> b = call.input( 2 );
        // We need to make sure that the matrices have a common/compatible layout,
        // ..before we can before the actual a @ b = c matrix multiplication!
        NDConfiguration.Layout layoutA = a.getNDConf().getLayout();
        NDConfiguration.Layout layoutB = b.getNDConf().getLayout();
        NDConfiguration.Layout layoutC = c.getNDConf().getLayout();

        boolean aIsCompatible = isRMOrCM( layoutA );
        boolean bIsCompatible = isRMOrCM( layoutB );

        Function relayout = Neureka.get().backend().getFunction().relayout();

        if ( aIsCompatible ) {
            if ( layoutB != NDConfiguration.Layout.SYMMETRIC )
                b = relayout.with(Arg.Layout.of(layoutA)).call(b); // We choose a valid layout based on a
            layoutC = layoutA;
        } else if ( bIsCompatible ) {
            if ( layoutA != NDConfiguration.Layout.SYMMETRIC )
                a = relayout.with(Arg.Layout.of(layoutB)).call(a); // We choose a valid layout based on b
            layoutC = layoutB;
        } else {
            // Ok so the inputs are unspecific/symmetric/ (not RM or CM)
            // So we just need to decide on any valid layout really:
            layoutC = isRMOrCM(layoutC) ? layoutC : NDConfiguration.Layout.ROW_MAJOR;
            a = relayout.with(Arg.Layout.of(layoutC)).call(a);
            b = relayout.with(Arg.Layout.of(layoutC)).call(b);
        }

        c.mut().toLayout( layoutC );
        c.mut().setIsVirtual( false ); // This statement is after the layout conversion for performance reasons (virtual tensors barely need copying).

        return call.withInputAt( 1, a ).withInputAt( 2, b );
    }

    private static boolean isRMOrCM(NDConfiguration.Layout layout ) {
        return layout == NDConfiguration.Layout.ROW_MAJOR ||
               layout == NDConfiguration.Layout.COLUMN_MAJOR;
    }

    /**
     *  This method will clone {@link Tensor} instances if they do not
     *  possess a simple {@link neureka.ndim.config.NDConfiguration}.
     *  This is usually the case when they are slices or permuted views on data...
     *  The reason for this is simply that we need inline data for the OpenCL kernels!
     *
     *
     * @param call The execution call whose tensors ought to be cloned based on the complexity of their access patterns.
     */
    private static ExecutionCall<?> _autoClone( ExecutionCall<?> call ) {
        for ( int i = 0; i < call.arity(); i++ )
            if (
                (!_isSimpleRowMajorMatrix( call.input( i ) ) && !_isSimpleColumnMajorMatrix( call.input( i ) ))
                        ||
                call.input( i ).isPartialSlice()
            ) {
                _LOG.debug("Auto cloning a tensor which does not have a simple ND configuration...");
                call = call.withInputAt( i, call.input( i ).deepCopy().mut().setIsIntermediate( true ) );
                /*
                    The user should do cloning explicitly because using slices
                    will cause the backend to perform auto cloning every time the
                    slice is being used for operations like this one...
                 */
            }

        return call;
    }

    private static boolean _isSimpleColumnMajorMatrix( Tensor<?> t ) {
        return t.rank() == 2 && t.getNDConf().getLayout() == NDConfiguration.Layout.COLUMN_MAJOR;
    }

    private static boolean _isSimpleRowMajorMatrix( Tensor<?> t ) {
        return t.rank() == 2 && t.getNDConf().getLayout() == NDConfiguration.Layout.ROW_MAJOR;
    }

}
