package neureka.dtype.custom;

import neureka.common.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.math.BigInteger;
import java.util.Iterator;

public final class UI64 extends AbstractNumericType<BigInteger, BigInteger[], Long, long[]>
{

    public UI64() { super(); }

    @Override public boolean signed() { return false; }

    @Override public int numberOfBytes() { return 8; }

    @Override public Class<BigInteger> targetType() { return BigInteger.class; }

    @Override
    public Class<BigInteger[]> targetArrayType() { return BigInteger[].class; }

    @Override
    public Class<Long> holderType() { return Long.class; }

    @Override
    public Class<long[]> holderArrayType() { return long[].class; }

    @Override
    public BigInteger foreignHolderBytesToTarget(byte[] bytes ) { // This is working but not optimal
        // use "import static java.math.BigInteger.ONE;" to shorten this line
        BigInteger UNSIGNED_LONG_MASK = BigInteger.ONE.shiftLeft(Long.SIZE).subtract(BigInteger.ONE);
        long unsignedLong = new BigInteger(bytes).longValue(); // sample input value
        BigInteger bi =  BigInteger.valueOf(unsignedLong).and(UNSIGNED_LONG_MASK);
        return bi;
    }

    @Override public BigInteger toTarget( Long original ) { return new BigInteger( Long.toUnsignedString( original ) ); }

    @Override
    public byte[] targetToForeignHolderBytes( BigInteger b ) {
        byte[] unsignedbyteArray= b.toByteArray();
        byte[] bytes = new byte[8];
        System.arraycopy(
                unsignedbyteArray, Math.max(0, unsignedbyteArray.length-8),
                bytes, 0,
                Math.min( 8, unsignedbyteArray.length )
        );
        return bytes;
    }

    @Override
    public BigInteger[] readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException {
        return new BigInteger[ 0 ];
    }

    @Override
    public <T> BigInteger[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        return new BigInteger[0];
    }

    @Override
    public long[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        return new long[0];
    }

    @Override
    public <T> long[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        return new long[0];
    }

    @Override
    public Long convertToHolder( Object from ) {
        return DataConverter.get().convert( from, Long.class );
    }

    @Override
    public long[] convertToHolderArray( Object from ) {
        return new long[0];
    }

    @Override
    public BigInteger convertToTarget( Object from ) {
        return null;
    }

    @Override
    public BigInteger[] convertToTargetArray( Object from ) {
        return DataConverter.get().convert( from, BigInteger[].class );
    }


}
