package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.nio.ByteBuffer;

public class F32 extends AbstractNumericType<Float, float[], Float, float[]>
{
    public F32() { super(); }

    @Override
    public boolean signed() {
        return true;
    }

    @Override
    public int numberOfBytes() {
        return 4;
    }

    @Override
    public Class<Float> targetType() {
        return Float.class;
    }

    @Override
    public Class<float[]> targetArrayType() {
        return float[].class;
    }

    @Override
    public Class<Float> foreignType() {
        return Float.class;
    }

    @Override
    public Class<float[]> foreignArrayType() {
        return float[].class;
    }

    @Override
    public Float foreignBytesToTarget(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getFloat();
    }

    @Override
    public Float toTarget(Float original) {
        return original;
    }

    @Override
    public byte[] targetToForeignBytes(Float number) {
        int intBits =  Float.floatToIntBits(number);
        return new byte[] {
                (byte) (intBits >> 24),
                (byte) (intBits >> 16),
                (byte) (intBits >> 8),
                (byte) (intBits)
        };
    }

    @Override
    public float[] readAndConvertDataFrom(DataInput stream, int size) throws IOException {
        return new float[ 0 ];
    }

    @Override
    public float[] readForeignDataFrom(DataInput stream, int size) throws IOException {
        return new float[0];
    }


}
