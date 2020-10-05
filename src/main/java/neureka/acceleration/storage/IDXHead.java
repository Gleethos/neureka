package neureka.acceleration.storage;


import neureka.Tsr;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.*;

import java.io.*;
import java.math.BigInteger;
import java.util.Map;
import java.util.stream.Collectors;

public class IDXHead implements FileHead
{
    private int _dataPointer;
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

    public IDXHead(String fileName)
    {
        _fileName = fileName;
        try {
            _construct(fileName);
        } catch(Exception e) {
            System.err.print("Failed reading IDX file!");
        }
    }

    private void _construct(String fileName) throws IOException {
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

        int zeros = numre.readIntegerInByteNumber((byte) 2);
        assert zeros == 0;

        int typeId = numre.readIntegerInByteNumber((byte) 1);
        Class<?> typeClass = TYPE_MAP.get(typeId);
        _dtype = DataType.instance(typeClass);

        int rank = numre.readIntegerInByteNumber((byte)1);
        int[] shape = new int[rank];

        int size = 1;
        for ( int i = 0; i < rank; i++ ) {
            shape[i] = numre.readIntegerInByteNumber((byte)4);
            size *= shape[i];
        }
        _shape = shape;
        _bodySize = size;

        _dataPointer = numre.bytesRead();

        //byte[] data = new byte[size];
        //assert f.read(data) == data.length;
        //f.close();
        //return data;
    }


    @Override
    public void persist(Tsr<?> t) throws IOException {

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

        f.write(new byte[]{0, 0});
        f.write(CODE_MAP.get(t.getValueClass()).byteValue());
        byte rank = (byte) t.rank();
        f.write(rank);
        int[] shape = t.getNDConf().shape();
        for ( int i = 0; i < rank; i++ ) {
            f.write(BigInteger.valueOf(shape[i]).toByteArray());
        }

        //Class<?> clazz = _dtype.getTypeClass();
        //if ( NumericType.class.isAssignableFrom(clazz) ) {
        //    NumericType<?,?> type = ((NumericType<?,?>)_dtype.getTypeClassInstance());
        //    DataInput stream = new DataInputStream(
        //            new BufferedInputStream(
        //                    fs,
        //                    _dataPointer + _bodySize * type.numberOfBytes()
        //            )
        //    );
        //    stream.skipBytes(_dataPointer);
        //    Object value = type.readDataFrom(
        //            stream,
        //            _bodySize
        //    );
        //    return new Tsr<>(_shape, _dtype, value);
        //}

    }

    @Override
    public Tsr<?> load() throws IOException {
        FileInputStream fs = new FileInputStream(_fileName);

        Class<?> clazz = _dtype.getTypeClass();
        if ( NumericType.class.isAssignableFrom(clazz) ) {
            NumericType<?,?> type = ((NumericType<?,?>)_dtype.getTypeClassInstance());
            DataInput stream = new DataInputStream(
                    new BufferedInputStream(
                            fs,
                            _dataPointer + _bodySize * type.numberOfBytes()
                    )
            );
            stream.skipBytes(_dataPointer);
            Object value = type.readDataFrom(
                    stream,
                    _bodySize
            );
            return new Tsr<>(_shape, _dtype, value);
        }
        return null;
    }
}
