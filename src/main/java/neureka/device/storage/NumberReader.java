package neureka.device.storage;

import neureka.dtype.NumericType;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;

public class NumberReader
{
    private final FileInputStream _fileInputStream;
    private int _bytesRead = 0;
    private final Map<Integer, byte[]> _byteMap = Map.of(
            1, new byte[1],
            2, new byte[2],
            4, new byte[4],
            8, new byte[8]
    );

    public NumberReader( FileInputStream fileInputStream ) {
        _fileInputStream = fileInputStream;
    }

    public FileInputStream getStream() {
        return _fileInputStream;
    }

    public <T> T read(NumericType<T, ?> type) throws IOException {
        assert _fileInputStream.read( _byteMap.get(type.numberOfBytes()) ) == type.numberOfBytes();
        _bytesRead += type.numberOfBytes();
        return type.convert(_byteMap.get(type.numberOfBytes()));
    }

    public Integer readIntegerInByteNumber(byte number) throws IOException {
        _bytesRead += number;
        assert _fileInputStream.read(_byteMap.get(number)) == number;
        return NumericType.Utility.unsignedByteArrayToInt(_byteMap.get(number));
    }

    public int bytesRead(){
        return _bytesRead;
    }

}
