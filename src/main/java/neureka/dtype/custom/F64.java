package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.nio.ByteBuffer;

public class F64 extends AbstractNumericType<Double, double[], Double, double[]>
{

    public F64() { super(); }

    @Override
    public boolean signed() {
        return true;
    }

    @Override
    public int numberOfBytes() {
        return 8;
    }

    @Override
    public Class<Double> targetType() {
        return Double.class;
    }

    @Override
    public Class<double[]> targetArrayType() {
        return double[].class;
    }

    @Override
    public Class<Double> foreignType() {
        return Double.class;
    }

    @Override
    public Class<double[]> foreignArrayType() {
        return double[].class;
    }

    @Override
    public Double foreignBytesToTarget(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getDouble();
    }

    @Override
    public Double toTarget(Double original) {
        return original;
    }

    @Override
    public byte[] targetToForeignBytes(Double number) {
        long data = Double.doubleToRawLongBits(number);
        return new byte[] {
                (byte) ((data >> 56) & 0xff),
                (byte) ((data >> 48) & 0xff),
                (byte) ((data >> 40) & 0xff),
                (byte) ((data >> 32) & 0xff),
                (byte) ((data >> 24) & 0xff),
                (byte) ((data >> 16) & 0xff),
                (byte) ((data >> 8) & 0xff),
                (byte) ((data >> 0) & 0xff),
        };
        //double d = 65.43;
        //byte[] output = new byte[8];
        //long lng = Double.doubleToLongBits(d);
        //for(int i = 0; i < 8; i++) output[i] = (byte)((lng >> ((7 - i) * 8)) & 0xff);
    }

    @Override
    public double[] readAndConvertDataFrom(DataInput stream, int size) throws IOException {
        return _readFrom( stream, size );
    }

    @Override
    public double[] readForeignDataFrom(DataInput stream, int size) throws IOException {
        return _readFrom( stream, size );
    }

    private double[] _readFrom( DataInput stream, int size ) throws IOException {
        double[] data = new double[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[ i ] = foreignBytesToTarget(_data);
        }
        return data;
    }

}
