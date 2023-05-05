package neureka.backend.main.operations;

import neureka.Shape;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *  Methods inside this utility class execute only some {@link ExecutionCall} arguments
 *  in groups if their total number exceeds the arity of an operation.
 *  
 */
public class ElemWiseUtil
{
    private static final Logger _LOG = LoggerFactory.getLogger( ElemWiseUtil.class );

    public static <V> Tsr<V> newTsrLike( Tsr<V> template, double value ) {
        return newTsrLike(
            template.itemType(),
            template.shape(),
            template.isOutsourced(),
            template.get( Device.class ),
            value
        );
    }

    public static <V> Tsr<V> newTsrLike(
        Class<V> type, Shape shape, boolean isOutsourced, Device<Object> device, double value
    ) {
        Tsr<V> t = Tsr.of( type, shape, value ).mut().setIsIntermediate( true );
        t.mut().setIsVirtual( false );
        t.mut().setItems( value );
        try {
            if ( isOutsourced ) device.store( t );
        } catch ( Exception exception ) {
            _LOG.error( "Failed storing a newly created tensor from a template tensor to its host device.", exception );
            throw exception;
        }
        return t;
    }

}
