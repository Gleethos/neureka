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


import neureka.common.utility.LogUtil;
import neureka.dtype.custom.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Constructor;
import java.util.*;

/**
 *  This class is a Multiton implementation for wrapping and representing type classes.
 *  Every {@link DataType} instance uniquely wraps a {@link Class} instance which will always differ
 *  from instances wrapped by other {@link DataType} instances.
 *  This is because the Multiton implementation utilizes a hash map where classes are the
 *  keys and their corresponding values are DataType instances.
 *
 * @param <Type> The type parameter of the type class whose instances ought to be represented.
*/
public final class DataType<Type>
{
    private final static int CAPACITY = 128;

    private static final Map<Class<?>, DataType> _instances = new LinkedHashMap<Class<?>, DataType>() {
        @Override
        protected boolean removeEldestEntry(final Map.Entry eldest) {
            return size() > CAPACITY;
        }
    };

    private final Logger _log;

    private final Class<Type> _typeClass;


    /**
     *  This method finds the corresponding NumericType implementation representing
     *  the passed type class or simply the provided class if no representation has been found.
     *
     * @param typeClass The type class whose "actual" / representation ought to be determined.
     * @return The true representation or simply itself if no NumericType representation has been found.
     */
    private static Class<?> _numericTypeRepresentationOf( Class<?> typeClass ) {
        Class<?> realTypeClass = typeClass; // The or case is for kotlin!
        if      ( typeClass == Double.class  || typeClass.getSimpleName().equals("double")) realTypeClass = F64.class;
        else if ( typeClass == Float.class   || typeClass.getSimpleName().equals("float") ) realTypeClass = F32.class;
        else if ( typeClass == Integer.class || typeClass.getSimpleName().equals("int")   ) realTypeClass = I32.class;
        else if ( typeClass == Short.class   || typeClass.getSimpleName().equals("short") ) realTypeClass = I16.class;
        else if ( typeClass == Long.class    || typeClass.getSimpleName().equals("long")  ) realTypeClass = I64.class;
        else if ( typeClass == Byte.class    || typeClass.getSimpleName().equals("byte")  ) realTypeClass = I8.class;
        return realTypeClass;
    }

    public static <T> DataType<T> of( Class<T> typeClass )
    {
        LogUtil.nullArgCheck(typeClass, "typeClass", Class.class);
        Class<?> realTypeClass = _numericTypeRepresentationOf( typeClass );

        if ( _instances.containsKey( realTypeClass ) ) {
            return _instances.get( realTypeClass );
        }
        DataType<T> dt = new DataType( realTypeClass );
        _instances.put( realTypeClass, dt );
        return dt;
    }

    private DataType( Class<Type> type ) {
        _typeClass = type;
        _log = LoggerFactory.getLogger(
                    DataType.class.getSimpleName() + ".of(" + _typeClass.getSimpleName() + ")"
                );
    }

    /**
     * @return An instance of the type class if possible.
     */
    public <T extends NumericType<?,?,?,?>> T getTypeClassInstance(Class<T> type)
    {
        if ( !type.isAssignableFrom(_typeClass) )
            throw new IllegalArgumentException("This data type does not support built in numeric type utilities!");
        Constructor<?>[] constructors = _typeClass.getDeclaredConstructors();
        Constructor<?> constructor = null;
        for ( Constructor<?> current : constructors ) {
            constructor = current;
            if (current.getGenericParameterTypes().length == 0)
                break;
        }

        try {
            constructor.setAccessible( true );
            return (T) constructor.newInstance();
        } catch ( Exception e ) {
            _log.error("Could not instantiate type class '"+ _typeClass.getSimpleName()+"': "+e.getMessage());
            e.printStackTrace();
        }
        return null;
    }

    /**
     * @param interfaceClass The type class which ought to be checked for compatibility.
     * @return True if the provided type is a sub-type of the type represented by this instance.
     */
    public boolean typeClassImplements( Class<?> interfaceClass ) {
        return interfaceClass.isAssignableFrom( _typeClass );
    }

    public Class<?> dataArrayType() {
        if ( this.typeClassImplements( NumericType.class ) )
            return ( (NumericType<?,?,?,?>) Objects.requireNonNull( getTypeClassInstance(NumericType.class) ) ).holderArrayType();
        else
            return Object[].class;
    }

    public <T> T virtualize( T value )
    {
        Object newValue;
        if ( _typeClass == F64.class )
            newValue = ( ( (double[]) value ).length <= 1 ) ? value : new double[]{ ( (double[]) value )[ 0 ] };
        else if ( _typeClass == F32.class )
            newValue = ( ( (float[]) value ).length <= 1 ) ? value : new float[]{ ( (float[]) value )[ 0 ] };
        else if ( _typeClass == I64.class )
            newValue = ( ( (long[]) value ).length <= 1 ) ? value : new long[]{ ( (long[]) value )[ 0 ] };
        else if ( _typeClass == I32.class )
            newValue = ( ( (int[]) value ).length <= 1 ) ? value : new int[]{ ( (int[]) value )[ 0 ] };
        else if ( _typeClass == I16.class )
            newValue = ( ( (short[]) value ).length <= 1 ) ? value : new short[]{ ( (short[]) value )[ 0 ] };
        else if ( _typeClass == I8.class )
            newValue = ( ( (byte[]) value ).length <= 1 ) ? value : new byte[]{ ( (byte[]) value )[ 0 ] };
        else
            newValue = ( ( (Object[]) value ).length <= 1 ) ? value : new Object[]{ ( (Object[]) value )[ 0 ] };

        return (T) newValue;
    }

    public boolean equals(final Object o) {
        if (o == this) return true;
        if (!(o instanceof DataType)) return false;
        final DataType<?> other = (DataType<?>) o;
        if ( !Objects.equals(_log, other._log) ) return false;
        final Object this$_typeClass = this.getRepresentativeType();
        final Object other$_typeClass = other.getRepresentativeType();
        if ( !Objects.equals(this$_typeClass, other$_typeClass) )
            return false;
        return true;
    }

    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final Object $_log = _log;
        result = result * PRIME + ($_log == null ? 43 : $_log.hashCode());
        final Object $_typeClass = this.getRepresentativeType();
        result = result * PRIME + ($_typeClass == null ? 43 : $_typeClass.hashCode());
        return result;
    }

    public String toString() {
        return "DataType[class=" + this.getRepresentativeType().getSimpleName() + "]";
    }

    public Class<?> getRepresentativeType() { return _typeClass; }

    public Class<Type> getValueTypeClass() {
        if ( this.typeClassImplements(NumericType.class) )
            return (this.getTypeClassInstance(NumericType.class)).holderType();
        else
            return _typeClass;
    }
}
