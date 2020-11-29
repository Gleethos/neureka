package neureka.dtype;

import neureka.dtype.custom.*;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

public abstract class AbstractNumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
implements NumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
{
    private static final Map<?,?> _NUMERIC_TYPE_RELATIONS = Map.of(
            I8.class, I8.class,
            I16.class, I16.class,
            I32.class, I32.class,
            I64.class, I64.class,
            F32.class, F32.class,
            F64.class, F64.class,
            UI8.class, I16.class,
            UI16.class, I32.class,
            UI64.class, UI64.class // think about this
    );

    public interface Conversion<FromType, ToType> { ToType go( FromType thing ); }

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
