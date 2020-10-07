package neureka.dtype;

import java.io.DataOutput;
import java.io.IOException;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public abstract class AbstractNumericType<TargetType, ArrayType> implements NumericType<TargetType, ArrayType>
{
    public interface Conversion<FromType, ToType> {
        ToType go(FromType thing);
    }

    private Map<Class<?>, Conversion<ArrayType,?>> _converters = new HashMap<>();

    protected byte[] _data;

    public AbstractNumericType() {
        _data = new byte[numberOfBytes()];
    }

    protected <T> void _set(
            Class<T> to,
            Conversion<ArrayType,T> conversion
    ){
        Map<Class<?>, Conversion<ArrayType,?>> fromMap = _converters;
        if ( !_converters.containsKey(to) )
        {
            _converters.put(to, conversion);
        } else {
            Conversion<?,?> found = fromMap.get(to);
            if ( found != null ) throw new IllegalStateException("Conversion already present!");
            else fromMap.put(to, conversion);
        }
    }

    @Override
    public <T> T convert(ArrayType from, Class<T> to) {
        return (T) _converters.get( to ).go( from );
    }

    @Override
    public void writeDataTo(DataOutput stream, Iterator<TargetType> iterator) throws IOException {
        while(iterator.hasNext()) {
            _data = convert(iterator.next());
            stream.write(_data);
        }
    }

}
