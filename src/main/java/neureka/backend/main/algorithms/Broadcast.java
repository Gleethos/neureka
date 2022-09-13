package neureka.backend.main.algorithms;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.ADActionSupplier;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.main.internal.RecursiveExecutor;
import neureka.backend.main.operations.other.Reshape;
import neureka.devices.Device;
import neureka.dtype.NumericType;

public final class Broadcast extends AbstractFunDeviceAlgorithm<Broadcast>
{
    public Broadcast( RecursiveExecutor finalExecutor )
    {
        super("broadcast");
        setIsSuitableFor(
            call ->
            {
                boolean isInvalid =
                            !call.validate()
                                 .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                                 .isValid();

                if ( isInvalid )
                    return SuitabilityPredicate.UNSUITABLE;

                int maxRank = 0;
                for ( Tsr<?> t : call.inputs() )
                    if ( t != null && t.rank() > maxRank ) maxRank = t.rank();

                for ( int i = 0; i < maxRank; i++ )
                {
                    int currentDim = -1;
                    for( Tsr<?> t : call.inputs() )
                    {
                        if ( t != null && i < t.rank() ) {
                            if ( currentDim == -1 ) currentDim = t.shape( i );
                            else if ( currentDim != t.shape( i ) && currentDim != 1 && t.shape( i ) != 1 ) return 0.0f;
                        }
                    }
                }
                return SuitabilityPredicate.GOOD;
            }
        );
        setAutogradModeFor(
            call ->
                call.validate()
                    .all( ( first, second ) -> first.shape().equals(second.shape()) )
                    .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                    .orElse(AutoDiffMode.BACKWARD_ONLY)
        );
        setDeviceExecution( (call, callback) -> finalExecutor.execute(call, callback), (ADActionSupplier) null );
        setCallPreparation(
            call ->
            {
                int offset = ( call.input( Number.class, 0 ) == null ? 1 : 0 );
                if (
                        call.input( Number.class, offset).shape().size() != call.input( Number.class, 1+offset).shape().size()
                )
                {
                    Tsr<?>[] inputs = {call.input( Number.class, offset), call.input( Number.class, 1+offset) };
                    Reshape.makeFit( inputs, true );
                    inputs = new Tsr[]{ null, inputs[0], inputs[1] };
                    call = call.withInputs( inputs );
                }

                Device device = call.getDevice();
                if ( call.input( 0 ) == null ) // Creating a new tensor:
                {
                    int[] s1 = call.input( 1 ).getNDConf().shape();
                    int[] s2 = call.input( 2 ).getNDConf().shape();

                    assert s1.length == s2.length;
                    int[] outShape = new int[s1.length];

                    for ( int i = 0; i < outShape.length; i++ )
                        assert s1[ i ] == 1 || s2[ i ] == 1 || s1[ i ] == s2[ i ];

                    for ( int i = 0; i < outShape.length; i++ )
                        outShape[ i ] = ( s1[ i ] == 1 ? s2[ i ] : s1[ i ] );

                    Class<Object> type = (Class<Object>) call.input(  1 ).getItemType();
                    Tsr<?> output = Tsr.of(type).withShape(outShape).all( 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    try {
                        device.store( output );
                    } catch( Exception e ) {
                        e.printStackTrace();
                    }
                    call = call.withInputAt( 0, output );
                }
                return call;
            }
        );
    }

}
