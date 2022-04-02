package neureka.devices.opencl;

import neureka.Neureka;
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
                Arrays.fill(newData, ((Number) (data)).floatValue());
                data = newData;
            }
            else if ( data instanceof Double ) {
                double[] newData = new double[size];
                Arrays.fill(newData, ((Number) (data)).doubleValue());
                data = newData;
            }
            // TODO: ...
        }

        boolean convertToFloat = Neureka.get().backend().get(CLContext.class).getSettings().isAutoConvertToFloat();
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
        }
        return data;
    }

    Pointer getPointer() {
        if ( _data instanceof float[] ) return Pointer.to((float[])_data);
        if ( _data instanceof double[] ) return Pointer.to((double[])_data);
        throw new IllegalStateException();
    }

    long getLength() {
        if ( _data instanceof float[] ) return ((float[])_data).length;
        if ( _data instanceof double[] ) return ((double[])_data).length;
        throw new IllegalStateException();
    }

    int getItemSize() {
        if ( _data instanceof float[] ) return Sizeof.cl_float;
        if ( _data instanceof double[] ) return Sizeof.cl_double;
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
        if ( o instanceof byte[] ) return ((byte[])o).length;
        if ( o instanceof long[] ) return ((long[])o).length;
        if ( o instanceof short[] ) return ((short[])o).length;
        throw new IllegalArgumentException();
    }

}
