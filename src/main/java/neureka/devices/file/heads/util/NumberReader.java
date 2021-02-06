package neureka.devices.file.heads.util;

import neureka.dtype.NumericType;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class NumberReader
{
    private final FileInputStream _fileInputStream;
    private int _bytesRead = 0;
    private final Map<Integer, byte[]> _byteMap;

    public NumberReader( FileInputStream fileInputStream ) {
        _byteMap = new HashMap<>();
        _byteMap.put( 1, new byte[ 1 ] );
        _byteMap.put( 2, new byte[ 2 ] );
        _byteMap.put( 4, new byte[ 4 ] );
        _byteMap.put( 8, new byte[ 8 ] );
        _fileInputStream = fileInputStream;
    }

    public FileInputStream getStream() {
        return _fileInputStream;
    }

    public <T> T read( NumericType<T, ?, ?, ?> type ) throws IOException {
        assert _fileInputStream.read( _byteMap.get(type.numberOfBytes()) ) == type.numberOfBytes();
        _bytesRead += type.numberOfBytes();
        return type.foreignHolderBytesToTarget(_byteMap.get(type.numberOfBytes()));
        // return NumericType.Utility.unsignedByteArrayToInt(_byteMap.get(number));
    }



    public int bytesRead() {
        return _bytesRead;
    }

}
