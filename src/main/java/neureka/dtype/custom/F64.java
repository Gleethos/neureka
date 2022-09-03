package neureka.dtype.custom;

import neureka.common.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

public final class F64 extends AbstractNumericType<Double, double[], Double, double[]>
{

    public F64() { super(); }

    @Override public boolean signed() { return true; }

    @Override public int numberOfBytes() { return 8; }

    @Override public Class<Double> targetType() { return Double.class; }

    @Override public Class<double[]> targetArrayType() { return double[].class; }

    @Override public Class<Double> holderType() { return Double.class; }

    @Override public Class<double[]> holderArrayType() { return double[].class; }

    @Override
    public Double foreignHolderBytesToTarget( byte[] bytes ) {
        return ByteBuffer.wrap(bytes).getDouble();
    }

    @Override
    public Double toTarget( Double original ) {
        return original;
    }

    @Override
    public byte[] targetToForeignHolderBytes( Double number ) {
        long data = Double.doubleToRawLongBits(number);
        return new byte[] {
                (byte) ((data >> 56) & 0xff),
                (byte) ((data >> 48) & 0xff),
                (byte) ((data >> 40) & 0xff),
                (byte) ((data >> 32) & 0xff),
                (byte) ((data >> 24) & 0xff),
                (byte) ((data >> 16) & 0xff),
                (byte) ((data >>  8) & 0xff),
                (byte) ((data >>  0) & 0xff),
        };
    }

    @Override
    public double[] readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException {
        return _readFrom( stream, size );
    }

    @Override
    public <T> double[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        double[] data = new double[ size ];
        for ( int i=0; i<size; i++ ) data[ i ] = convertToTarget( iterator.next() );
        return data;
    }

    @Override
    public double[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        return _readFrom( stream, size );
    }

    @Override
    public <T> double[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        double[] data = new double[ size ];
        for ( int i=0; i<size; i++ ) data[ i ] = convertToHolder( iterator.next() );
        return data;
    }

    @Override
    public Double convertToHolder( Object from ) {
        if ( Byte.class.equals( from.getClass() ) )
            return ( (Byte) from ).doubleValue();
        else if ( Integer.class.equals( from.getClass() ) )
            return ( (Integer) from ).doubleValue();
        else if ( Double.class.equals( from.getClass() ) )
            return ( (Double) from );
        else if ( Short.class.equals( from.getClass() ) )
            return ( (Short) from ).doubleValue();
        else if ( Long.class.equals( from.getClass() ) )
            return ( (Long) from ).doubleValue();
        else if ( Float.class.equals( from.getClass() ) )
            return ( (Float) from ).doubleValue();
        else
            return null;
    }

    @Override
    public double[] convertToHolderArray( Object from ) {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public Double convertToTarget( Object from ) {
        return convertToHolder( from );
    }

    @Override
    public double[] convertToTargetArray( Object from ) {
        return DataConverter.get().convert( from, double[].class );
    }

    private double[] _readFrom( DataInput stream, int size ) throws IOException {
        double[] data = new double[ size ];
        byte[] bytes = new byte[ this.numberOfBytes() ];
        for ( int i=0; i<size; i++ ) {
            stream.readFully( bytes );
            data[ i ] = foreignHolderBytesToTarget( bytes );
        }
        return data;
    }

}
