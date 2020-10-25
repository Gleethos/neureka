package neureka.dtype;

import java.lang.reflect.Constructor;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.function.Consumer;

public class DataType
{
    private static Map<Class<?>, DataType> _instances = new WeakHashMap<>();

    public static DataType instance( Class<?> c ) {
        if ( _instances.containsKey(c) ) {
            return _instances.get( c );
        }
        DataType dt = new DataType(c);
        _instances.put(c, dt);
        return dt;
    }

    public static <T> void forType(Class<T> c, Consumer<T> action) {
        if ( _instances.containsKey(c) ) action.accept(
                (T) _instances.get(c)
        );
    }

    private Class<?> _type;

    private DataType(Class<?> type) {
        _type = type;
    }

    public Class getTypeClass(){
        return _type;
    }

    public Object getTypeClassInstance(){

        Constructor[] ctors = _type.getDeclaredConstructors();
        Constructor ctor = null;
        for (int i = 0; i < ctors.length; i++) {
            ctor = ctors[ i ];
            if (ctor.getGenericParameterTypes().length == 0)
                break;
        }

        try {
            ctor.setAccessible( true );
            return ctor.newInstance();
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        return null;
    }

    public boolean typeClassImplements(Class<?> interfaceClass){
        return interfaceClass.isAssignableFrom(_type);
    }

}
