package neureka.devices.file.handles;

import neureka.Tsr;
import neureka.devices.Storage;
import neureka.devices.file.FileHandle;
import org.slf4j.Logger;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @param <C> The concrete type extending this class.
 * @param <V> The value type of the data in the file represented by this class.
 */
public abstract class AbstractFileHandle<C, V> implements FileHandle<C, V>
{
    protected static Logger _LOG;

    private final String _extension;
    protected final String _fileName;

    protected int _size = 0;

    AbstractFileHandle( String filename, FileType type )
    {
        _fileName = filename;
        _extension = type.defaultExtension();
        if ( _fileName.equals( "" ) ) {
            String message = "Loading tensor from '" + extension() + "' file failed because the provided file location string is empty!\n";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        if ( !_fileName.contains( "." ) ) {
            String message = "Loading tensor from location '" + _fileName + "' failed because the file does not have an ending." +
                    "Expected file extension of type '" + extension() + "'!\n";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        String[] split = _fileName.split( "\\." ); // Example: splitting "myFile.PNG" int "myFile" and "PNG".
        String ending = split[ split.length-1 ].toLowerCase(); // ... 'ending' would then be "png"!
        if ( !ending.contains( extension().toLowerCase() ) ) {
            String message = "Loading tensor from location '"+_fileName+"' failed because the file ending does not match '"+extension()+"'!\n";
            _LOG.error( message );
            throw new IllegalArgumentException(message);
        }
    }

    public int size() { return _size; }

    public boolean isEmpty() { return _size == 0; }

    public boolean contains( Tsr<V> o ) {
        throw new IllegalStateException("Not supported!");
    }


    protected abstract Object _loadData() throws IOException;

    protected File _loadFile()
    {
            File found = new File( _fileName );
            if ( !found.exists() ) {
                String message = "Failed loading file at '"+_fileName+"' of type '"+extension()+"'!\n" +
                        "It seems like the file does not exist.\n";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
            return found;
    }

    protected FileInputStream _loadFileInputStream() throws IOException
    {
        File found = _loadFile();
        FileInputStream f = null;
        try
        {
            f = new FileInputStream( found );
        }
        catch ( FileNotFoundException e )
        {
            String message = "Could not create 'FileInputStream' for '"+found.toString()+"'.";
            _LOG.error( message, e );
            throw new IOException( message );
        }
        return f;
    }


    @Override
    public C free() {
        boolean success = new File(_fileName).delete();
        if ( !success ) {
            String message = "Freeing "+extension()+" file '"+_fileName+"' failed!\n";
            _LOG.error( message );
            throw new IllegalStateException( message );
        } else _size --;
        return (C) this;
    }

    @Override
    public String getLocation() { return _fileName; }

    @Override
    public String getFileName() {
        String[] split = _fileName.replace( "\\","/" ).split( "/" );
        return split[ split.length - 1 ];
    }

    @Override
    public Storage<V> restore( Tsr<V> tensor ) {
        try {
            Object value = _loadData();
            tensor.setItems( value );
        } catch ( Exception e ) {
            _LOG.error( "Restoring tensor from filesystem failed!\n", e );
            e.printStackTrace();
        }
        return this;
    }

    @Override public final String extension() { return _extension; }

}
