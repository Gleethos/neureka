package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tsr;
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
import neureka.ndim.config.types.simple.Simple1DConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DotProductAlgorithm extends AbstractFunDeviceAlgorithm<DotProductAlgorithm>
{
    static Logger _LOG = LoggerFactory.getLogger(DotProductAlgorithm.class);

    public DotProductAlgorithm() {
        super("dot_algorithm");
        setIsSuitableFor(
            call -> call.validate()
                    .allNotNull( t -> Number.class.isAssignableFrom(t.getItemType()) )
                    .allNotNull( t -> t.shape().count( d -> d > 1 ) <= 1 )
                    .getEstimator()
                    .goodIfAnyNonNull( t -> t.getNDConf() instanceof Simple1DConfiguration)
                    .getEstimation() * 1.1f
        );
        setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY );
        setExecution(
            (function, call) -> {
                call = _prepare( call );
                return
                    Result.of(AbstractDeviceAlgorithm.executeDeviceAlgorithm( call ))
                    .withAutoDiff( (Function f, ExecutionCall<? extends Device<?>> adCall ) ->
                    {
                        if ( adCall.autogradMode().allowsForward() )
                            throw new IllegalArgumentException("Dot product does not support forward-AD!");
                        Function mul = Neureka.get().backend().getFunction().mul();
                        int d = ( 1 + adCall.getValOf( Arg.DerivIdx.class ) ) % 2;
                        Tsr<?> derivative = adCall.input( d ).T().deepCopy().mut().setIsIntermediate( true ); // We need to clone it to make it have a simple nd configuration...
                        derivative.to(adCall.getDevice());
                        return ADAction.of( target -> mul.execute( target.error(), derivative ) );
                    });
            }
        );
        setCallPreparation( c -> c );
    }


    private static ExecutionCall<Device<Object>> _prepare( ExecutionCall call )
    {
        assert call.arity() <= 3;
        if ( call.arity() == 2 ) call = call.withAddedInputAt(0, null);

        call = _withDimTrim( call );

        if ( call.input( 0 ) == null ) // Creating a new tensor:
            call = _withNewOutput( call );

        return (ExecutionCall<Device<Object>>) _autoClone( call );
    }

    private static ExecutionCall<?> _withDimTrim( ExecutionCall<?> call ) {
        Tsr<?> a = call.input( 0 );
        Tsr<?> b = call.input( 1 );
        Tsr<?> c = call.input( 2 );
        Function dimTrim = Neureka.get().backend().getAutogradFunction().dimTrim();
        if ( a != null && a.rank() > 1 ) call = call.withInputAt( 0, dimTrim.execute( a ) );
        if ( b != null && b.rank() > 1 ) call = call.withInputAt( 1, dimTrim.execute( b ) );
        if ( c != null && c.rank() > 1 ) call = call.withInputAt( 2, dimTrim.execute( c ) );
        return call;
    }

    private static ExecutionCall<?> _withNewOutput( ExecutionCall<?> call )
    {
        Class<Number> type = (Class<Number>) call.input(  1 ).getDataType().getItemTypeClass();

        Tsr<Number> output = Tsr.of( type ).withShape( 1 ).all( 0 ).mut().setIsIntermediate( true );

        NDConfiguration.Layout layoutOut = _checkAndPrepareLayout( call.input( 1 ), call.input( 2 ), output );
        output.mut().toLayout( layoutOut );
        output.mut().setIsVirtual( false ); // This statement is after the layout conversion for performance reasons (virtual tensors barely need copying).

        call.getDeviceFor(Number.class).store( output );
        return call.withInputAt( 0, output );
    }

    private static NDConfiguration.Layout _checkAndPrepareLayout(Tsr<?> a, Tsr<?> b, Tsr<?> c )
    {
        // We need to make sure that the vectors have a common/compatible layout,
        // ..before we can do the actual a . b = c dot product!
        NDConfiguration.Layout layoutA = a.getNDConf().getLayout();
        NDConfiguration.Layout layoutB = b.getNDConf().getLayout();
        NDConfiguration.Layout layoutC = c.getNDConf().getLayout();

        boolean aIsCompatible = isSymmetric( layoutA );
        boolean bIsCompatible = isSymmetric( layoutB );

        if ( aIsCompatible ) {
            b.mut().toLayout(layoutA); // We choose a valid layout based on a
            layoutC = layoutA;
        } else if ( bIsCompatible ) {
            a.mut().toLayout(layoutB); // We choose a valid layout based on b
            layoutC = layoutB;
        } else {
            // Ok so the inputs are unspecific (or RM or CM)
            // So we just need to decide on any valid layout really:
            layoutC = isSymmetric(layoutC) ? layoutC : NDConfiguration.Layout.SYMMETRIC;
            a.mut().toLayout( layoutC );
            b.mut().toLayout( layoutC );
        }

        return layoutC;
    }

    private static boolean isSymmetric( NDConfiguration.Layout layout ) {
        return layout == NDConfiguration.Layout.SYMMETRIC;
    }

    /**
     *  This method will clone {@link Tsr} instances if they do not
     *  possess a simple {@link neureka.ndim.config.NDConfiguration}.
     *  This is usually the case when they are slices or permuted views on data...
     *  The reason for this is simply that we need inline data for the OpenCL/CPU kernels!
     *
     * @param call The execution call whose tensors ought to be cloned based on the complexity of their access patterns.
     */
    private static ExecutionCall<?> _autoClone( ExecutionCall<?> call ) {
        for (int i = 0; i < call.arity(); i++ ) {
            if (
                    !_isSimpleSymmetric( call.input( i ) )
                            ||
                    call.input( i ).isPartialSlice()
            ) {
                _LOG.debug("Auto cloning a tensor which does not have a simple symmetric ND configuration...");
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

    private static boolean _isSimpleSymmetric( Tsr<?> t ) {
        return t.rank() == 1 && t.getNDConf().getLayout() == NDConfiguration.Layout.SYMMETRIC;
    }

}
