package neureka.dtype;

import neureka.dtype.custom.*;

import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public abstract class AbstractNumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
implements NumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
{
    private static final Map<?,?> _relations = Map.of(
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

    private final Map<Class<?>, Conversion<Object, ?>> _arrayTargetConverters = new HashMap<>();
    private final Map<Class<?>, Conversion<Object, ?>> _arrayHolderConverters = new HashMap<>();
    private final Map<Class<?>, Conversion<Object,?>> _scalarTargetConverters = new HashMap<>();
    private final Map<Class<?>, Conversion<Object,?>> _scalarHolderConverters = new HashMap<>();


    protected byte[] _data;

    public AbstractNumericType() {
        _data = new byte[numberOfBytes()];
    }

    protected <T> void _setToTarget( Class<T> from, Conversion<T,TargetType> conversion ) {
        _setConversion( _scalarTargetConverters, from, (Conversion<Object, ?>) conversion);
    }

    protected <T> void _setToHolder( Class<T> from, Conversion<T,HolderType> conversion ) {
        _setConversion( _scalarHolderConverters, from, (Conversion<Object, ?>) conversion);
    }

    protected <T> void _setToTargetArray( Class<T> from, Conversion<T,TargetArrayType> conversion ) {
        _setConversion( _arrayTargetConverters, from, (Conversion<Object, ?>) conversion);
    }

    protected <T> void _setToHolderArray( Class<T> from, Conversion<T,HolderArrayType> conversion ) {
        _setConversion( _arrayHolderConverters, from, (Conversion<Object, ?>) conversion);
    }

    private static void _setConversion(
            Map<Class<?>, Conversion<Object, ?>> converters, Class<?> key, Conversion<Object,?> conversion
    ){
        if ( !converters.containsKey(key) ) converters.put( key, conversion );
        else {
            Conversion<?,?> found = converters.get( key );
            if ( found != null ) throw new IllegalStateException("Conversion already present!");
            else converters.put( key, conversion );
        }
    }


    @Override
    public Class<NumericType<TargetType, TargetArrayType, TargetType, TargetArrayType>> getNumericTypeTarget() {
        return (Class<NumericType<TargetType, TargetArrayType, TargetType, TargetArrayType>>) _relations.get( this.getClass() );
    }

    @Override
    public void writeDataTo( DataOutput stream, Iterator<TargetType> iterator ) throws IOException {
        while( iterator.hasNext() ) {
            _data = targetToForeignHolderBytes(iterator.next());
            stream.write(_data);
        }
    }

}
