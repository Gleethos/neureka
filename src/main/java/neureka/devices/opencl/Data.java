package neureka.devices.opencl;

import neureka.Neureka;
import neureka.backend.api.BackendContext;
import neureka.common.utility.DataConverter;
import org.jocl.Pointer;
import org.jocl.Sizeof;

import java.util.Arrays;

public class Data {

    private final Object _data;

    public static Data of( Object data ) {
        return new Data( data, 0, lengthOf(data) );
    }

    public static Data of( Object data, int start, int size ) {
        return new Data( data, start, size );
    }

    private Data( Object data, int start, int size ) {
        _data = _preprocess( data, start, size );
    }

    Object getData() { return _data; }

    private Object _preprocess(Object data, int start, int size) {

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
        boolean clContextFound = backend.has(CLContext.class);
        boolean convertToFloat = clContextFound && backend.get(CLContext.class).getSettings().isAutoConvertToFloat();
        if ( convertToFloat ) // NOTE: Currently we only support floats!
            data = DataConverter.get().convert(data, float[].class);

        // TODO: Enable this for more types:
        if ( data instanceof float[] ) {
            float[] array = (float[]) data;
            if (start > 0) {
                float[] newData = new float[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof double[] ) {
            double[] array = (double[]) data;
            if (start > 0) {
                double[] newData = new double[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof int[] ) {
            int[] array = (int[]) data;
            if (start > 0) {
                int[] newData = new int[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof long[] ) {
            long[] array = (long[]) data;
            if (start > 0) {
                long[] newData = new long[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof short[] ) {
            short[] array = (short[]) data;
            if (start > 0) {
                short[] newData = new short[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        } else if ( data instanceof byte[] ) {
            byte[] array = (byte[]) data;
            if (start > 0) {
                byte[] newData = new byte[size];
                System.arraycopy(array, start, newData, 0, newData.length);
                data = newData;
            }
        }
        return data;
    }

    Pointer getPointer() {
        if ( _data instanceof float[] ) return Pointer.to((float[])_data);
        if ( _data instanceof double[] ) return Pointer.to((double[])_data);
        if ( _data instanceof int[] ) return Pointer.to((int[])_data);
        if ( _data instanceof short[] ) return Pointer.to((short[])_data);
        if ( _data instanceof long[] ) return Pointer.to((long[])_data);
        if ( _data instanceof byte[] ) return Pointer.to((byte[])_data);
        throw new IllegalStateException();
    }

    long getLength() {
        if ( _data instanceof float[] ) return ((float[])_data).length;
        if ( _data instanceof double[] ) return ((double[])_data).length;
        if ( _data instanceof int[] ) return ((int[])_data).length;
        if ( _data instanceof short[] ) return ((short[])_data).length;
        if ( _data instanceof long[] ) return ((long[])_data).length;
        if ( _data instanceof byte[] ) return ((byte[])_data).length;
        throw new IllegalStateException();
    }

    int getItemSize() {
        if ( _data instanceof float[] ) return Sizeof.cl_float;
        if ( _data instanceof double[] ) return Sizeof.cl_double;
        if ( _data instanceof int[] ) return Sizeof.cl_int;
        if ( _data instanceof short[] ) return Sizeof.cl_short;
        if ( _data instanceof long[] ) return Sizeof.cl_long;
        if ( _data instanceof byte[] ) return 1;
        throw new IllegalStateException();
    }

    OpenCLDevice.cl_dtype getType() {
        if ( _data instanceof float[] ) return OpenCLDevice.cl_dtype.F32;
        if ( _data instanceof double[] ) return OpenCLDevice.cl_dtype.F64;
        throw new IllegalStateException();
    }

    private static int lengthOf( Object o ) {
        if ( o instanceof Number ) return 1;
        if ( o instanceof float[] ) return ((float[])o).length;
        if ( o instanceof double[] ) return ((double[])o).length;
        if ( o instanceof int[] ) return ((int[])o).length;
        if ( o instanceof long[] ) return ((long[])o).length;
        if ( o instanceof short[] ) return ((short[])o).length;
        if ( o instanceof byte[] ) return ((byte[])o).length;
        throw new IllegalArgumentException();
    }

}
