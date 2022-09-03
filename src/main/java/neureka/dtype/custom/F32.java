package neureka.dtype.custom;

import neureka.common.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

public final class F32 extends AbstractNumericType<Float, float[], Float, float[]>
{
    public F32() {
        super();
    }

    @Override public boolean signed() { return true; }

    @Override public int numberOfBytes() { return 4; }

    @Override public Class<Float> targetType() { return Float.class; }

    @Override public Class<float[]> targetArrayType() { return float[].class; }

    @Override public Class<Float> holderType() { return Float.class; }

    @Override public Class<float[]> holderArrayType() { return float[].class; }

    @Override
    public Float foreignHolderBytesToTarget( byte[] bytes ) {
        return ByteBuffer.wrap(bytes).getFloat();
    }

    @Override
    public Float toTarget(Float original) {
        return original;
    }

    @Override
    public byte[] targetToForeignHolderBytes(Float number) {
        int intBits =  Float.floatToIntBits(number);
        return new byte[] {
                (byte) (intBits >> 24),
                (byte) (intBits >> 16),
                (byte) (intBits >> 8),
                (byte) (intBits)
        };
    }

    @Override
    public float[] readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public <T> float[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        float[] data = new float[size];
        for ( int i=0; i<size; i++ ) data[ i ] = convertToHolder( iterator.next() );
        return data;
    }


    @Override
    public float[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public <T> float[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        return readAndConvertForeignDataFrom( iterator, size );
    }

    @Override
    public Float convertToHolder( Object from )
    {
        if ( Byte.class.equals( from.getClass() ) )
            return ( (Byte) from ).floatValue();
        else if ( Integer.class.equals( from.getClass() ) )
            return ( (Integer) from ).floatValue();
        else if ( Double.class.equals( from.getClass() ) )
            return ( (Double) from ).floatValue();
        else if ( Short.class.equals( from.getClass() ) )
            return ( (Short) from ).floatValue();
        else if ( Long.class.equals( from.getClass() ) )
            return ( (Long) from ).floatValue();
        else if ( Float.class.equals( from.getClass() ) )
            return ( (Float) from );
        else
            return null;
    }

    @Override
    public float[] convertToHolderArray( Object from ) {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public Float convertToTarget( Object from ) {
        return convertToHolder( from );
    }

    @Override
    public float[] convertToTargetArray( Object from ) {
        return DataConverter.get().convert( from, float[].class );
    }


}
