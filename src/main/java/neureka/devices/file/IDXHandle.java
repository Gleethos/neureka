package neureka.devices.file;



import neureka.Neureka;
import neureka.Tsr;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.*;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.Collectors;

/**
 *  This class is one of many extensions of the {@link AbstractFileHandle} which
 *  is therefore ultimately an implementation of the {@link FileHandle} interface.
 *  Like other {@link FileHandle} implementations this class represents a file
 *  of a given type, in this case it represents a IDX file.
*/
public final class IDXHandle extends AbstractFileHandle<IDXHandle, Number>
{
    static {
        _LOG = LoggerFactory.getLogger( IDXHandle.class );
    }
    private DataType<?> _dataType;
    private int _dataOffset;
    private int _valueSize;
    private int[] _shape;

    private static final Map<Integer, Class<?>> TYPE_MAP;
    static {
        TYPE_MAP = new HashMap<>();
        TYPE_MAP.put( 0x08, UI8.class );  // unsigned byte
        TYPE_MAP.put( 0x09, I8.class  );  // signed byte
        TYPE_MAP.put( 0x0A, UI16.class);  //-> !! This is speculation !!
        TYPE_MAP.put( 0x0B, I16.class );  // short (2 bytes)
        TYPE_MAP.put( 0x0C, I32.class );  // int (4 bytes)
        TYPE_MAP.put( 0x0D, F32.class );  // float (4 bytes)
        TYPE_MAP.put( 0x0E, F64.class );  // double (8 bytes)
    }

    private final static Map<Class<?>, Integer> CODE_MAP = TYPE_MAP.entrySet()
                                                            .stream()
                                                            .collect(
                                                                Collectors.toMap(
                                                                    Map.Entry::getValue,
                                                                    Map.Entry::getKey
                                                                )
                                                            );

    public IDXHandle(String fileName)
    {
        super( fileName, new IDXType() );
        try {
            _loadHead();
        } catch( Exception e ) {
            e.printStackTrace();
            System.err.print("Failed reading IDX file!");
        }
    }

    public IDXHandle(Tsr<Number> t, String filename ) {
        super( filename, new IDXType() );
        _shape = t.getNDConf().shape();
        _dataType = t.getDataType();
        t.getMut().setIsVirtual( false );
        store( t );
    }

    private void _loadHead() throws IOException
    {
        FileInputStream f = _loadFileInputStream();

        NumberReader numre = new NumberReader( f );

        int zeros = numre.read( new UI16() );
        assert zeros == 0;

        int typeId = numre.read( new UI8() );
        Class<?> typeClass = TYPE_MAP.get( typeId );
        _dataType = DataType.of( typeClass );

        int rank = numre.read( new UI8() );
        int[] shape = new int[ rank ];

        int size = 1;
        for ( int i = 0; i < rank; i++ ) {
            shape[ i ] = numre.read( new UI32() ).intValue();
            size *= shape[ i ];
        }

        _shape = shape;
        _valueSize = size;
        _dataOffset = numre.bytesRead();
    }


    @Override
    public <T extends Number> IDXHandle store(Tsr<T> tensor )
    {
        Iterator<T> data = tensor.iterator();
        FileOutputStream fos;
        try
        {
            fos = new FileOutputStream(_fileName);
        }
        catch (FileNotFoundException e)
        {
            try {
                File newFile = new File( _fileName );
                fos = new FileOutputStream( newFile );
            } catch ( Exception innerException ) {
                innerException.printStackTrace();
                return this;
            }
        }
        BufferedOutputStream f = new BufferedOutputStream(fos);

        int offset = 0;

        try {
            f.write( new byte[]{ 0, 0 } );
            offset += 2;
            f.write( CODE_MAP.get( _dataType.getRepresentativeType() ).byteValue() );
            offset += 1;
            byte rank = (byte) _shape.length;
            f.write( rank );
            offset += 1;
            int bodySize = 1;
            for ( int i = 0; i < rank; i++ ) {
                byte[] integer = ByteBuffer.allocate( 4 ).putInt( _shape[ i ] ).array();
                assert integer.length == 4;
                f.write(integer);
                bodySize *= _shape[ i ];
                offset += 4;
            }
            _dataOffset = offset;
            _valueSize = bodySize;
            NumericType<Number, Object, Number, Object> type = (NumericType<Number, Object, Number, Object>) _dataType.getTypeClassInstance(NumericType.class);

            type.writeDataTo( new DataOutputStream( f ), (Iterator<Number>) data );
            f.close();
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        return this;
    }

    @Override
    protected Object _loadData() throws IOException {
        FileInputStream fs = new FileInputStream( _fileName );
        Class<?> clazz = _dataType.getRepresentativeType();
        if ( NumericType.class.isAssignableFrom( clazz ) ) {
            NumericType<?,?,?,?> type = _dataType.getTypeClassInstance(NumericType.class);
            DataInput stream = new DataInputStream(
                    new BufferedInputStream( fs, _dataOffset + _valueSize * type.numberOfBytes() )
            );
            stream.skipBytes( _dataOffset );
            if ( Neureka.get().settings().dtype().getIsAutoConvertingExternalDataToJVMTypes() )
                return type.readAndConvertForeignDataFrom( stream, _valueSize);
            else
                return type.readForeignDataFrom( stream, _valueSize);
        }
        return null;
    }

    @Override
    public Tsr<Number> load() throws IOException
    {
        Object value = _loadData();
        DataType<?> type = Neureka.get().settings().dtype().getIsAutoConvertingExternalDataToJVMTypes()
                            ? DataType.of( _dataType.getTypeClassInstance(NumericType.class).getNumericTypeTarget() )
                            : _dataType;
        return Tsr.of( type, _shape, value ).getMut().upcast(Number.class);
    }

    @Override
    public int getDataSize() {
        int bytes = ( _dataType.typeClassImplements( NumericType.class ) )
                ? _dataType.getTypeClassInstance(NumericType.class).numberOfBytes()
                : 1;
        return _valueSize * bytes;
    }

    @Override
    public int getTotalSize() {
        return getDataSize() + _dataOffset;
    }

    public DataType<?> getDataType() {
        return _dataType;
    }

    public int getValueSize() {
        return _valueSize;
    }

    public int[] getShape() {
        return _shape;
    }

    private static class IDXType implements FileType
    {
        @Override public String defaultExtension() { return "idx"; }
    }
}
