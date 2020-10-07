package neureka.acceleration.storage;

import neureka.dtype.NumericType;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;

public class NumberReader
{
    private FileInputStream _fileInputStream;
    private byte[] _oneByte = new byte[1];
    private byte[] _twoBytes = new byte[2];
    private byte[] _fourBytes = new byte[4];
    private int _bytesRead = 0;
    private Map<Integer, byte[]> _byteMap = Map.of(
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
        if ( number == 1 ) {
            assert _fileInputStream.read( _oneByte ) == 1;
            _bytesRead ++;
            return NumericType.Utility.unsignedByteArrayToInt( _oneByte );
        } else if ( number == 2 ) {
            _bytesRead += 2;
            assert _fileInputStream.read( _twoBytes ) == 2;
            return NumericType.Utility.unsignedByteArrayToInt( _twoBytes );
        } else if ( number == 4 ) {
            _bytesRead += 4;
            assert _fileInputStream.read( _fourBytes ) == 4;
            return NumericType.Utility.unsignedByteArrayToInt( _fourBytes );
        }
        return 0;
    }

    public int bytesRead(){
        return _bytesRead;
    }

}
