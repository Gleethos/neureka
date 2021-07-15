/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   _____        _     _______
  |  __ \      | |   |__   __|
  | |  | | __ _| |_ __ _| |_   _ _ __   ___
  | |  | |/ _` | __/ _` | | | | | '_ \ / _ \
  | |__| | (_| | || (_| | | |_| | |_) |  __/
  |_____/ \__,_|\__\__,_|_|\__, | .__/ \___|
                            __/ | |
                           |___/|_|

 */

package neureka.dtype;


import neureka.dtype.custom.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Constructor;
import java.util.Arrays;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.function.Consumer;

/**
 *  This class is a Multiton implementation for wrapping and representing type classes.
 *  Every DataType instance uniquely wraps a Class instance which will always differ
 *  from instances wrapped by other DataType instances.
 *  This is because the Multiton implementation utilizes a hash map where classes are the
 *  keys and their corresponding values are DataType instances.
 *
 *
 * @param <Type> The type parameter of the type class whose instances ought to be represented.
*/
public final class DataType<Type>
{
    private static final Map<Class<?>, DataType> _instances = new WeakHashMap<>();

    /**
     *  This method finds the corresponding NumericType implementation representing
     *  the passed type class or simply the provided class if no representation has been found.
     *
     * @param typeClass The type class whose "actual" / representation ought to be determined.
     * @return The true representation or simply itself if no NumericType representation has been found.
     */
    private static Class<?> _numericTypeRepresentationOf(Class<?> typeClass ) {
        Class<?> realTypeClass = typeClass;
        if ( typeClass == Double.class ) realTypeClass = F64.class;
        else if ( typeClass == Float.class ) realTypeClass = F32.class;
        else if ( typeClass == Integer.class ) realTypeClass = I32.class;
        else if ( typeClass == Short.class ) realTypeClass = I16.class;
        else if ( typeClass == Long.class ) realTypeClass = I64.class;
        else if ( typeClass == Byte.class ) realTypeClass = I8.class;
        else if ( typeClass == byte[].class ) realTypeClass = I8.class;
        else if ( typeClass == int[].class ) realTypeClass = I32.class;
        else if ( typeClass == float[].class ) realTypeClass = F32.class;
        else if ( typeClass == double[].class ) realTypeClass = F64.class;
        else if ( typeClass == long[].class ) realTypeClass = I64.class;
        return realTypeClass;
    }

    public static <T> DataType<T> of(Class<T> typeClass )
    {
        Class<?> realTypeClass = _numericTypeRepresentationOf( typeClass );

        if ( _instances.containsKey( realTypeClass ) ) {
            return _instances.get( realTypeClass );
        }
        DataType<T> dt = new DataType( realTypeClass );
        _instances.put( realTypeClass, dt );
        return dt;
    }

    public static <T> void forType( Class<T> typeClass, Consumer<DataType<T>> action )
    {
        Class<?> realTypeClass = _numericTypeRepresentationOf( typeClass );
        if ( _instances.containsKey( realTypeClass ) ) {
            DataType<?> found = _instances.get( realTypeClass );
            if ( found.getTypeClass() == typeClass ) action.accept( (DataType<T>) found );
        }
    }

    private final Logger _log;

    private final Class<Type> _typeClass;

    private DataType( Class<Type> type ) {
        _typeClass = type;
        _log = LoggerFactory.getLogger(
                DataType.class.getSimpleName() + ".of(" + _typeClass.getSimpleName() + ")"
        );
    }

    /**
     * @return An instance of the type class if possible.
     */
    public Type getTypeClassInstance()
    {
        Constructor<?>[] constructors = _typeClass.getDeclaredConstructors();
        Constructor<?> constructor = null;
        for ( Constructor<?> current : constructors ) {
            constructor = current;
            if (current.getGenericParameterTypes().length == 0)
                break;
        }

        try {
            constructor.setAccessible( true );
            return (Type) constructor.newInstance();
        } catch ( Exception e ) {
            _log.error("Could not instantiate type class '"+ _typeClass.getSimpleName()+"': "+e.getMessage());
            e.printStackTrace();
        }
        return null;
    }

    public boolean typeClassImplements( Class<?> interfaceClass ) {
        return interfaceClass.isAssignableFrom(_typeClass);
    }


    public <T> T virtualize(T value )
    {
        Object newValue;
        if ( getTypeClass() == F64.class )
            newValue = ( ( (double[]) value ).length <= 1 ) ? value : new double[]{ ( (double[]) value )[ 0 ] };
        else if ( getTypeClass() == F32.class )
            newValue = ( ( (float[]) value ).length <= 1 ) ? value : new float[]{ ( (float[]) value )[ 0 ] };
        else if ( getTypeClass() == I32.class )
            newValue = ( ( (int[]) value ).length <= 1 ) ? value : new int[]{ ( (int[]) value )[ 0 ] };
        else if ( getTypeClass() == I16.class )
            newValue = ( ( (short[]) value ).length <= 1 ) ? value : new short[]{ ( (short[]) value )[ 0 ] };
        else if ( getTypeClass() == I8.class )
            newValue = ( ( (byte[]) value ).length <= 1 ) ? value : new byte[]{ ( (byte[]) value )[ 0 ] };
        else
            newValue = ( ( (Object[]) value ).length <= 1 ) ? value : new Object[]{ ( (Object[]) value )[ 0 ] };

        return (T) newValue;
    }

    public <T> T actualize(T value, int size )
    {
        Object newValue = value;
        if ( getTypeClass() == F64.class ) {
            if ( ( (double[]) value ).length == size ) return value;
            newValue = new double[ size ];
            Arrays.fill( (double[]) newValue, ( (double[]) value )[ 0 ] );
        } else if ( getTypeClass() == F32.class ) {
            if ( ( (float[]) value ).length == size ) return value;
            newValue = new float[size];
            Arrays.fill( (float[]) newValue, ( (float[]) value )[ 0 ] );
        } else if ( getTypeClass() == I32.class ) {
            if ( ( (int[]) value ).length == size ) return value;
            newValue = new int[ size ];
            Arrays.fill( (int[]) newValue, ( (int[]) value )[ 0 ] );
        } else if ( getTypeClass() == I16.class ) {
            if ( ( (short[]) value ).length == size ) return value;
            newValue = new short[ size ];
            Arrays.fill( (short[]) newValue, ( (short[]) value )[ 0 ] );
        } else if ( getTypeClass() == I8.class ) {
            if ( ( (byte[]) value ).length == size ) return value;
            newValue = new byte[ size ];
            Arrays.fill( (byte[]) newValue, ( (byte[]) value )[ 0 ] );
        } else if ( getTypeClass() == I64.class ) {
            if ( ( (long[]) value ).length == size ) return value;
            newValue = new long[ size ];
            Arrays.fill( (long[]) newValue, ( (long[]) value )[ 0 ] );
        } else {
            if ( ( (Object[]) value ).length == size ) return value;
            newValue = new Object[ size ];
            Arrays.fill( (Object[]) newValue, ( (Object[]) value )[ 0 ] );
        }
        return (T) newValue;
    }

    public Object allocate( int size )
    {
        if ( getTypeClass() == F64.class )
            return new double[ size ];
        else if ( getTypeClass() == F32.class )
            return new float[ size ];
        else if ( getTypeClass() == I32.class || getTypeClass() == UI32.class )
            return new int[ size ];
        else if ( getTypeClass() == I16.class || getTypeClass() == UI16.class )
            return new short[ size ];
        else if ( getTypeClass() == I8.class || getTypeClass() == UI8.class )
            return new byte[ size ];
        else if ( getTypeClass() == I64.class || getTypeClass() == UI64.class )
            return new long[ size ];
        else
            return new Object[ size ];
    }


    public Logger getLog() {
        return this._log;
    }

    public boolean equals(final Object o) {
        if (o == this) return true;
        if (!(o instanceof DataType)) return false;
        final DataType<?> other = (DataType<?>) o;
        final Object this$_log = this.getLog();
        final Object other$_log = other.getLog();
        if (this$_log == null ? other$_log != null : !this$_log.equals(other$_log)) return false;
        final Object this$_typeClass = this.getTypeClass();
        final Object other$_typeClass = other.getTypeClass();
        if (this$_typeClass == null ? other$_typeClass != null : !this$_typeClass.equals(other$_typeClass))
            return false;
        return true;
    }

    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final Object $_log = this.getLog();
        result = result * PRIME + ($_log == null ? 43 : $_log.hashCode());
        final Object $_typeClass = this.getTypeClass();
        result = result * PRIME + ($_typeClass == null ? 43 : $_typeClass.hashCode());
        return result;
    }

    public String toString() {
        return "DataType(" + this.getTypeClass() + ")";
    }

    public Class<Type> getTypeClass() {
        return this._typeClass;
    }

    public Class<?> getJVMTypeClass() {
        if ( this.typeClassImplements(NumericType.class) )
            return ((NumericType) this.getTypeClassInstance()).holderType();

        return getTypeClass();
    }
}
