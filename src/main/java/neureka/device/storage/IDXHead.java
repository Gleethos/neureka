package neureka.device.storage;


import neureka.Tsr;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.*;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.Collectors;

public class IDXHead implements FileHead
{
    private int _dataOffset;
    private int _bodySize;
    private String _fileName;
    private DataType _dtype;
    private int[] _shape;

    private static Map<Integer, Class<?>> TYPE_MAP =  Map.of(
            0x08, UI8.class,   // unsigned byte
            0x09, I8.class,    // signed byte
            0x0A, UI16.class,  //-> !! This is speculation !!
            0x0B, I16.class,   // short (2 bytes)
            0x0C, I32.class,   // int (4 bytes)
            0x0D, F32.class,   // float (4 bytes)
            0x0E, F64.class    // double (8 bytes)
    );

    private final static Map<Class<?>, Integer> CODE_MAP = TYPE_MAP.entrySet()
                                                        .stream()
                                                        .collect(
                                                                Collectors.toMap(
                                                                        Map.Entry::getValue,
                                                                        Map.Entry::getKey
                                                                )
                                                        );

    public IDXHead( String fileName )
    {
        _fileName = fileName;
        try {
            _load(fileName);
        } catch(Exception e) {
            System.err.print("Failed reading IDX file!");
        }
    }

    public IDXHead( Tsr<?> t, String filename ) throws IOException {
        _fileName = filename;
        _shape = t.getNDConf().shape();
        _dtype = t.getDataType();
        t.setIsVirtual(false);
        persist(t.iterator());
    }

    private void _load(String fileName) throws IOException
    {
        FileInputStream f = null;
        try
        {
            f = new FileInputStream(fileName);
        }
        catch (FileNotFoundException e)
        {
            //System.err.println("File: " + fileName + " not found.");
            return; // This mean that the file will be created when tensor is saved...
        }
        NumberReader numre = new NumberReader(f);

        int zeros = numre.read( new UI16() );
        assert zeros == 0;

        int typeId = numre.read( new UI8() );
        Class<?> typeClass = TYPE_MAP.get(typeId);
        _dtype = DataType.instance(typeClass);

        int rank = numre.read( new UI8() );
        int[] shape = new int[rank];

        int size = 1;
        for ( int i = 0; i < rank; i++ ) {
            shape[i] = numre.read( new UI32() ).intValue();
            size *= shape[i];
        }


        _shape = shape;
        _bodySize = size;

        _dataOffset = numre.bytesRead();

        //byte[] data = new byte[size];
        //assert f.read(data) == data.length;
        //f.close();
        //return data;
    }


    @Override
    public <T> void persist( Iterator<T> data ) throws IOException
    {
        FileOutputStream fos;
        try
        {
            fos = new FileOutputStream(_fileName);
        }
        catch (FileNotFoundException e)
        {
            fos = new FileOutputStream(new File(_fileName));
        }
        BufferedOutputStream f = new BufferedOutputStream(fos);

        int offset = 0;

        f.write(new byte[]{0, 0});
        offset += 2;
        f.write( CODE_MAP.get( _dtype.getTypeClass() ).byteValue() );
        offset += 1;
        byte rank = (byte) _shape.length;
        f.write(rank);
        offset += 1;
        int bodySize = 1;
        for ( int i = 0; i < rank; i++ ) {
            byte[] integer = ByteBuffer.allocate(4).putInt(_shape[i]).array();
            assert integer.length == 4;
            f.write(integer);
            bodySize *= _shape[i];
            offset += 4;
        }
        _dataOffset = offset;
        _bodySize = bodySize;
        NumericType<T,Object> type = ( NumericType<T, Object> ) _dtype.getTypeClassInstance();

        type.writeDataTo( new DataOutputStream(f), data);
        f.close();

    }

    @Override
    public Tsr<?> load() throws IOException
    {
        FileInputStream fs = new FileInputStream(_fileName);
        Class<?> clazz = _dtype.getTypeClass();
        if ( NumericType.class.isAssignableFrom(clazz) ) {
            NumericType<?,?> type = ((NumericType<?,?>)_dtype.getTypeClassInstance());
            DataInput stream = new DataInputStream(
                    new BufferedInputStream(
                            fs,
                            _dataOffset + _bodySize * type.numberOfBytes()
                    )
            );
            stream.skipBytes(_dataOffset);
            Object value = type.readDataFrom(
                    stream,
                    _bodySize
            );
            return new Tsr<>(_shape, _dtype, value);
        }
        return null;
    }
}
