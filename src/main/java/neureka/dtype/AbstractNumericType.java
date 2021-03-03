package neureka.dtype;

import neureka.dtype.custom.*;

import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 *  This class is a common precursor class for the concrete implementation of the "NumericType" interface (see documentation).
 *  It implements shared logic which will be used by these concrete classes.
 *  This is especially true for the target/holder type relationship between the numeric types.
 *
 * @param <TargetType> The target type is the targeted JVM data-type which can represent the holder type.
 * @param <TargetArrayType> The target array type is the targeted JVM array data-type which can represent the holder array type.
 * @param <HolderType> The holder type is the JVM type which can hold the data but not necessarily represent it (int cant hold uint).
 * @param <HolderArrayType> The holder array type is the JVM array type which can hold the data but not necessarily represent it (int[] cant hold uint[]).
 */
public abstract class AbstractNumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
implements NumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
{
    private static final Map<Object,Object> _NUMERIC_TYPE_RELATIONS;
    static {
        /*
         *  The following map stores the representative relationships between concrete numeric type implementations.
         *  For example, the unsigned integer type can be represented by the JVM without information loss
         *  by the signed long type...
         */
        _NUMERIC_TYPE_RELATIONS = new HashMap<>();
        _NUMERIC_TYPE_RELATIONS.put( I8.class, I8.class    );
        _NUMERIC_TYPE_RELATIONS.put( I16.class, I16.class  );
        _NUMERIC_TYPE_RELATIONS.put( I32.class, I32.class  );
        _NUMERIC_TYPE_RELATIONS.put( I64.class, I64.class  );
        _NUMERIC_TYPE_RELATIONS.put( F32.class, F32.class  );
        _NUMERIC_TYPE_RELATIONS.put( F64.class, F64.class  );
        _NUMERIC_TYPE_RELATIONS.put( UI8.class, I16.class  );
        _NUMERIC_TYPE_RELATIONS.put( UI16.class, I32.class );
        _NUMERIC_TYPE_RELATIONS.put( UI64.class, UI64.class); // think about this
    }

    @Override
    public Class<NumericType<TargetType, TargetArrayType, TargetType, TargetArrayType>> getNumericTypeTarget() {
        return (Class<NumericType<TargetType, TargetArrayType, TargetType, TargetArrayType>>) _NUMERIC_TYPE_RELATIONS.get( this.getClass() );
    }

    @Override
    public void writeDataTo( DataOutput stream, Iterator<TargetType> iterator ) throws IOException {
        byte[] data;
        while( iterator.hasNext() ) {
            data = targetToForeignHolderBytes( iterator.next() );
            stream.write( data );
        }
    }

}
