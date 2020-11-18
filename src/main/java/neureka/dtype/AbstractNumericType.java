package neureka.dtype;

import neureka.dtype.custom.*;

import java.io.DataOutput;
import java.io.IOException;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public abstract class AbstractNumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
implements NumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
{
    private final Map<?,?> _relations = Map.of(
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

    public interface Conversion<FromType, ToType> { ToType go(FromType thing); }

    private Map<Class<?>, Conversion<TargetArrayType,?>> _converters = new HashMap<>();

    protected byte[] _data;

    public AbstractNumericType() {
        _data = new byte[numberOfBytes()];
    }

    protected <T> void _set(
            Class<T> to,
            Conversion<TargetArrayType,T> conversion
    ){
        Map<Class<?>, Conversion<TargetArrayType,?>> fromMap = _converters;
        if ( !_converters.containsKey(to) )
        {
            _converters.put(to, conversion);
        } else {
            Conversion<?,?> found = fromMap.get(to);
            if ( found != null ) throw new IllegalStateException("Conversion already present!");
            else fromMap.put(to, conversion);
        }
    }

    public Class<NumericType<TargetType, TargetArrayType, TargetType, TargetArrayType>> getJVMType() {
        return (Class<NumericType<TargetType, TargetArrayType, TargetType, TargetArrayType>>) _relations.get( this.getClass() );
    }

    @Override
    public <T> T convert(TargetArrayType from, Class<T> to) {
        return (T) _converters.get( to ).go( from );
    }

    @Override
    public void writeDataTo(DataOutput stream, Iterator<TargetType> iterator) throws IOException {
        while( iterator.hasNext() ) {
            _data = convert(iterator.next());
            stream.write(_data);
        }
    }

}
