package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.devices.Device;
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
                Result.of(AbstractDeviceAlgorithm.executeFor(
                    outerCaller, outerCall,
                    innerCall -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( innerCall )
                ))
                .withAutoDiff( (Function f, ExecutionCall<? extends Device<?>> adCall ) ->
                {
                    if ( adCall.autogradMode().allowsForward() )
                        throw new IllegalArgumentException("Matrix multiplication does not support forward-AD!");
                    Function matMul = Neureka.get().backend().getFunction().matMul();
                    int d = ( 1 + adCall.getValOf( Arg.DerivIdx.class ) ) % 2;
                    Tsr<?> derivative = adCall.input( d ).T().deepCopy().mut().setIsIntermediate( true ); // We need to clone it to make it have a simple nd configuration...
                    derivative.to(adCall.getDevice());
                    return ADAction.of(target ->
                            d == 1
                                    ? matMul.execute( target.error(), derivative )
                                    : matMul.execute( derivative, target.error() )
                    );
                })
        );
        setCallPreparation(MatMulAlgorithm::_prepare);
    }


    private static ExecutionCall<Device<Object>> _prepare( ExecutionCall call )
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
        Tsr<Number> output = Tsr.of( type ).withShape( shp ).all( 0 ).mut().setIsIntermediate( true );

        NDConfiguration.Layout layoutOut = _checkAndPrepareLayout( call.input( 1 ), call.input( 2 ), output );
        output.mut().toLayout( layoutOut );
        output.mut().setIsVirtual( false ); // This statement is after the layout conversion for performance reasons (virtual tensors barely need copying).

        call.getDeviceFor(Number.class).store( output );
        return call.withInputAt( 0, output );
    }

    private static NDConfiguration.Layout _checkAndPrepareLayout(Tsr<?> a, Tsr<?> b, Tsr<?> c )
    {
        // We need to make sure that the matrices have a common/compatible layout,
        // ..before we can before the actual a @ b = c matrix multiplication!
        NDConfiguration.Layout layoutA = a.getNDConf().getLayout();
        NDConfiguration.Layout layoutB = b.getNDConf().getLayout();
        NDConfiguration.Layout layoutC = c.getNDConf().getLayout();

        boolean aIsCompatible = isRMOrCM( layoutA );
        boolean bIsCompatible = isRMOrCM( layoutB );

        if ( aIsCompatible ) {
            b.mut().toLayout(layoutA); // We choose a valid layout based on a
            layoutC = layoutA;
        } else if ( bIsCompatible ) {
            a.mut().toLayout(layoutB); // We choose a valid layout based on b
            layoutC = layoutB;
        } else {
            // Ok so the inputs are unspecific/symmetric/ (not RM or CM)
            // So we just need to decide on any valid layout really:
            layoutC = isRMOrCM(layoutC) ? layoutC : NDConfiguration.Layout.ROW_MAJOR;
            a.mut().toLayout( layoutC );
            b.mut().toLayout( layoutC );
        }

        return layoutC;
    }

    private static boolean isRMOrCM(NDConfiguration.Layout layout ) {
        return layout == NDConfiguration.Layout.ROW_MAJOR ||
               layout == NDConfiguration.Layout.COLUMN_MAJOR;
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
                call = call.withInputAt( i, call.input( i ).deepCopy().mut().setIsIntermediate( true ) );
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

}
