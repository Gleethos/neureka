package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.math.BigInteger;

public class UI64 extends AbstractNumericType<BigInteger, BigInteger[]>
{

    public UI64(){ super(); }

    @Override
    public boolean signed() {
        return false;
    }

    @Override
    public int numberOfBytes() {
        return 8;
    }

    @Override
    public Class<BigInteger> targetType() {
        return BigInteger.class;
    }

    @Override
    public Class<BigInteger[]> targetArrayType() {
        return BigInteger[].class;
    }

    @Override
    public BigInteger convert(byte[] bytes) {
        return null;
    }

    @Override
    public byte[] convert(BigInteger number) {
        return new byte[0];
    }

    @Override
    public BigInteger[] readDataFrom(DataInput stream, int size) throws IOException {
        return new BigInteger[0];
    }

    @Override
    public void writeDataTo(DataOutput stream, BigInteger[] data) throws IOException {

    }


}
