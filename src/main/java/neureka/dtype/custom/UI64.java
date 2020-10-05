package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.math.BigInteger;
import java.nio.ByteBuffer;

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
        // use "import static java.math.BigInteger.ONE;" to shorten this line
        BigInteger UNSIGNED_LONG_MASK = BigInteger.ONE.shiftLeft(Long.SIZE).subtract(BigInteger.ONE);
        long unsignedLong = new BigInteger(bytes).longValue(); // sample input value
        BigInteger bi =  BigInteger.valueOf(unsignedLong).and(UNSIGNED_LONG_MASK);
        return bi;
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
