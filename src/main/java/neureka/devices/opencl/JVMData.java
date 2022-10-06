package neureka.devices.opencl;

import neureka.Neureka;
import neureka.backend.api.BackendContext;
import neureka.common.utility.DataConverter;
import org.jocl.Pointer;
import org.jocl.Sizeof;

import java.util.Arrays;

/**
 *  This defines a representation of some basic primitive numeric array based JVM data
 *  which may be stored on an {@link OpenCLDevice} eventually.
 *  <br> <br>
 *  <b>Warning: This is an internal class, meaning it should not be used
 *  anywhere but within this library. <br>
 *  This class or its public methods might change or get removed in future versions!</b>
 */
class JVMData
{
    private final Object _data;

    public static JVMData of( Class<?> type, int size ) {
        Object data = null;
        if      ( type == Float.class   ) data = new float[size];
        else if ( type == Double.class  ) data = new double[size];
        else if ( type == Integer.class ) data = new int[size];
        else if ( type == Long.class    ) data = new long[size];
        else if ( type == Short.class   ) data = new short[size];
        else if ( type == Byte.class    ) data = new byte[size];
        else if ( type == Boolean.class ) data = new boolean[size];
        else {
            String message = "Unsupported data type  '"+type+"' was encountered.\n";
            throw new IllegalArgumentException(message);
        }
        return of(data);
    }

    public static JVMData of( Object data ) {
        return new JVMData( data, 0, lengthOf(data) );
    }

    public static JVMData of( Object data, int size, int start ) {
        return new JVMData( data, start, size );
    }

    private JVMData( Object data, int start, int size ) {
        _data = _preprocess( data, start, size );
    }

    Object getArray() { return _data; }

    private Object _preprocess( Object data, int start, int size )
    {
        if ( data instanceof Number ) {
            if ( data instanceof Float ) {
                float[] newData = new float[size];
                Arrays.fill( newData, ((Float) (data)) );
                data = newData;
            } else if ( data instanceof Double ) {
                double[] newData = new double[size];
                Arrays.fill( newData, ((Double) (data)) );
                data = newData;
            } else if ( data instanceof Integer ) {
                int[] newData = new int[size];
                Arrays.fill( newData, ((Integer) (data)) );
                data = newData;
            } else if ( data instanceof Short ) {
                short[] newData = new short[size];
                Arrays.fill( newData, ((Short) (data)) );
                data = newData;
            } else if ( data instanceof Byte ) {
                byte[] newData = new byte[size];
                Arrays.fill( newData, ((Byte) (data)) );
                data = newData;
            } else if ( data instanceof Long ) {
                long[] newData = new long[size];
                Arrays.fill( newData, ((Long) (data)) );
                data = newData;
            }
        }

        BackendContext backend = Neureka.get().backend();
        boolean clContextFound = backend.has(CLBackend.class);
        boolean convertToFloat = clContextFound && backend.get(CLBackend.class).getSettings().isAutoConvertToFloat();
        if ( convertToFloat ) // NOTE: Currently we only support floats!
            data = DataConverter.get().convert(data, float[].class);

        // TODO: Enable this for more types:
        if ( data instanceof float[] ) {
            float[] array = (float[]) data;
            if ( start > 0 || size < array.length ) {
                float[] newData = new float[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof double[] ) {
            double[] array = (double[]) data;
            if ( start > 0 || size < array.length ) {
                double[] newData = new double[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof int[] ) {
            int[] array = (int[]) data;
            if ( start > 0 || size < array.length ) {
                int[] newData = new int[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof long[] ) {
            long[] array = (long[]) data;
            if ( start > 0 || size < array.length ) {
                long[] newData = new long[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof short[] ) {
            short[] array = (short[]) data;
            if ( start > 0 || size < array.length ) {
                short[] newData = new short[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof byte[] ) {
            byte[] array = (byte[]) data;
            if ( start > 0 || size < array.length ) {
                byte[] newData = new byte[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        }
        return data;
    }

    Pointer getPointer() {
        if ( _data instanceof float[]  ) return Pointer.to( (float[])  _data );
        if ( _data instanceof double[] ) return Pointer.to( (double[]) _data );
        if ( _data instanceof int[]    ) return Pointer.to( (int[])    _data );
        if ( _data instanceof short[]  ) return Pointer.to( (short[])  _data );
        if ( _data instanceof long[]   ) return Pointer.to( (long[])   _data );
        if ( _data instanceof byte[]   ) return Pointer.to( (byte[])   _data );
        throw new IllegalStateException();
    }

    long getLength() {
        if ( _data instanceof float[]  ) return ( (float[])  _data ).length;
        if ( _data instanceof double[] ) return ( (double[]) _data ).length;
        if ( _data instanceof int[]    ) return ( (int[])    _data ).length;
        if ( _data instanceof short[]  ) return ( (short[])  _data ).length;
        if ( _data instanceof long[]   ) return ( (long[])   _data ).length;
        if ( _data instanceof byte[]   ) return ( (byte[])   _data ).length;
        throw new IllegalStateException();
    }

    int getItemSize() {
        if ( _data instanceof float[]  ) return Sizeof.cl_float;
        if ( _data instanceof double[] ) return Sizeof.cl_double;
        if ( _data instanceof int[]    ) return Sizeof.cl_int;
        if ( _data instanceof short[]  ) return Sizeof.cl_short;
        if ( _data instanceof long[]   ) return Sizeof.cl_long;
        if ( _data instanceof byte[]   ) return 1;
        throw new IllegalStateException();
    }

    OpenCLDevice.cl_dtype getType() {
        if ( _data instanceof float[]  ) return OpenCLDevice.cl_dtype.F32;
        if ( _data instanceof double[] ) return OpenCLDevice.cl_dtype.F64;
        if ( _data instanceof int[]    ) return OpenCLDevice.cl_dtype.I32;
        if ( _data instanceof short[]  ) return OpenCLDevice.cl_dtype.I16;
        if ( _data instanceof long[]   ) return OpenCLDevice.cl_dtype.I64;
        if ( _data instanceof byte[]   ) return OpenCLDevice.cl_dtype.I8;
        throw new IllegalStateException();
    }

    Number getElementAt(int i) {
        if ( _data instanceof float[]  ) return ( (float[])  _data )[i];
        if ( _data instanceof double[] ) return ( (double[]) _data )[i];
        if ( _data instanceof int[]    ) return ( (int[])    _data )[i];
        if ( _data instanceof short[]  ) return ( (short[])  _data )[i];
        if ( _data instanceof long[]   ) return ( (long[])   _data )[i];
        if ( _data instanceof byte[]   ) return ( (byte[])   _data )[i];
        throw new IllegalStateException();
    }

    private static int lengthOf( Object o ) {
        if ( o instanceof Number   ) return 1;
        if ( o instanceof float[]  ) return ( (float[])  o ).length;
        if ( o instanceof double[] ) return ( (double[]) o ).length;
        if ( o instanceof int[]    ) return ( (int[])    o ).length;
        if ( o instanceof long[]   ) return ( (long[])   o ).length;
        if ( o instanceof short[]  ) return ( (short[])  o ).length;
        if ( o instanceof byte[]   ) return ( (byte[])   o ).length;
        throw new IllegalArgumentException();
    }

}
