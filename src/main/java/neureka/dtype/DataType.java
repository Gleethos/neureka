package neureka.dtype;

import neureka.Component;
import neureka.Tsr;

import java.util.Map;
import java.util.WeakHashMap;

public class DataType<ValueType> implements Component<Tsr<ValueType>>
{
    private static Map<Class, DataType> _instances = new WeakHashMap<>();

    public static DataType instance( Class c ) {
        if ( _instances.containsKey(c) ) {
            return _instances.get( c );
        }
        DataType dt = new DataType(c);
        _instances.put(c, dt);
        return dt;
    }

    private Class _type;

    private DataType(Class type) {
        _type = type;
    }

    public Class getTypeClass(){
        return _type;
    }

    @Override
    public void update(Tsr<ValueType> oldOwner, Tsr<ValueType> newOwner) {

    }

}
