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

import java.lang.reflect.Constructor;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.function.Consumer;

public class DataType<Type>
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

    private Class<Type> _type;

    private DataType(Class<Type> type) {
        _type = type;
    }

    public Class<Type> getTypeClass(){
        return _type;
    }

    public Type getTypeClassInstance(){

        Constructor[] ctors = _type.getDeclaredConstructors();
        Constructor ctor = null;
        for (int i = 0; i < ctors.length; i++) {
            ctor = ctors[ i ];
            if (ctor.getGenericParameterTypes().length == 0)
                break;
        }

        try {
            ctor.setAccessible( true );
            return (Type) ctor.newInstance();
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        return null;
    }

    public boolean typeClassImplements(Class<?> interfaceClass){
        return interfaceClass.isAssignableFrom(_type);
    }

}
