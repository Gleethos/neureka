package neureka.devices.storage;

import neureka.Tsr;
import neureka.devices.host.HostCPU;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.I16;
import neureka.dtype.custom.I8;
import neureka.dtype.custom.UI8;
import neureka.utility.DataConverter;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;

public class JPEGHead implements FileHead<JPEGHead, Number>
{
    private final String _fileName;

    int _width;
    int _height;
    //int _totalSize;

    public JPEGHead( String fileName )
    {
        _fileName = fileName;
        try {
            _loadHead( fileName );
        } catch( Exception e ) {
            System.err.print("Failed reading JPG file!");
        }
    }

    public JPEGHead( Tsr<Number> t, String filename ) {
        assert t.rank() == 3;
        assert t.shape( 2 ) == 3;
        _fileName = filename;
        _height = t.shape(0);
        _width = t.shape(1);
        //_shape = t.getNDConf().shape();
        //_dtype = t.getDataType();
        assert t.rank() == 3;
        assert t.shape(2) == 3;
        t.setIsVirtual( false );
        store( t );
    }


    private void _loadHead( String fileName )
    {
        File found = new File( fileName );
        if ( !found.exists() ) return; // No file exists (yet)!

        BufferedImage image = null;
        Raster data;
        try
        {
            image = ImageIO.read( found );
            data = image.getData();
            _height = data.getHeight();
            _width = data.getWidth();
        }
        catch ( IOException e )
        {
            e.printStackTrace();
        }
    }

    @Override
    public Tsr<Number> load() throws IOException {
        Object value = _loadData();
        Tsr t = new Tsr( new int[]{_height, _width, 3} );
        t.setValue( value );
        return t;
    }

    private Object _loadData()
    {
        File found = new File( _fileName );
        if ( !found.exists() ) return null; // Throw exception+

        BufferedImage image = null;
        try
        {
            image = ImageIO.read( found );
            byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            short[] newData = new short[ data.length ];

            UI8 ui8 = new UI8();
            //ui8.readAndConvertDataFrom( image.getRaster().getDataBuffer(), _height * _width * 3 );
            HostCPU.instance().getExecutor().threaded(
                    data.length,
                    ( start, end ) -> {
                        for ( int i=start; i<end; i++ ) newData[i] = ui8.toTarget( data[i] );
                    }
            );
            //Raster data = image.getData();
            //short[] value = new short[  _height * _width * 3 ]; //(H, W, D)
            //data.getPixels( 0, 0, _width, _height, value );
            return newData;//DataConverter.instance().convert( data, short[].class );
           // return value;
        }
        catch ( IOException e )
        {
            e.printStackTrace();
        }

        return null;
    }

    @Override
    public JPEGHead free() {
        boolean success = new File(_fileName).delete();
        if ( !success ) {
            System.err.println( "Freeing jpg file '"+_fileName+"' failed!" );
        }
        return this;
    }

    @Override
    public int getValueSize() {
        return _width * _height * 3;
    }

    @Override
    public int getDataSize() {
        return _width * _height * 3;
    }

    @Override
    public int getTotalSize() {
        return _width * _height * 3;
    }

    @Override
    public String getLocation() {
        return _fileName;
    }

    @Override
    public String getFileName() {
        String[] split = _fileName.replace( "\\","/" ).split( "/" );
        return split[ split.length - 1 ];
    }

    @Override
    public DataType<?> getDataType() {
        return DataType.instance( UI8.class );
    }

    @Override
    public int[] getShape() {
        return new int[]{ _height, _width, 3 };
    }

    @Override
    public Storage store( Tsr<Number> tensor )
    {
        //Iterator<Number> data = tensor.iterator();
        byte[] data = DataConverter.instance().convert( tensor.getData(), byte[].class );
        //BufferedImage buffi = new BufferedImage( _width, _height, BufferedImage.TYPE_3BYTE_BGR );
        //buffi.getRaster().
        BufferedImage buffi = new BufferedImage( _width, _height, BufferedImage.TYPE_3BYTE_BGR );
        buffi.setData(
                Raster.createRaster(
                        buffi.getSampleModel(), new DataBufferByte( data, data.length ),
                        new Point()
                )
        );
        try {
            //BufferedImage image = ImageIO.read( new ByteArrayInputStream( data ) );
            ImageIO.write( buffi, "jpg", new File( _fileName ) );
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        //System.out.println("image created");

        return this;
    }

    @Override
    public Storage restore( Tsr<Number> tensor ) {
        try {
            Object value = _loadData();
            tensor.setValue( value );
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        return this;
    }
}
