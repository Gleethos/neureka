package neureka.devices.storage;

import org.slf4j.Logger;

import java.io.File;

public abstract class AbstractFileHead<FinalType> implements FileHead<FinalType, Number>
{
    protected static Logger _LOGGER;

    protected final String _fileName;

    AbstractFileHead( String filename )
    {
        _fileName = filename;
        if ( _fileName.equals("") ) {
            String message = "Loading tensor from '"+extension()+"' file failed because the provided file location string is empty!";
            _LOGGER.error( message );
            throw new IllegalArgumentException(message);
        }
        if ( !_fileName.contains(".") ) {
            String message = "Loading tensor from location '"+_fileName+"' failed because the file does not have an ending." +
                    "Expected file extension of type '"+extension()+"'!";
            _LOGGER.error( message );
            throw new IllegalArgumentException(message);
        }
        String[] split = _fileName.split("\\."); // Example: splitting "myFile.PNG" int "myFile" and "PNG".
        String ending = split[ split.length-1 ].toLowerCase(); // ... 'ending' would then be "png"!
        if ( !ending.contains( extension().toLowerCase() ) ) {
            String message = "Loading tensor from location '"+_fileName+"' failed because the file ending does not match '"+extension()+"'!";
            _LOGGER.error( message );
            throw new IllegalArgumentException(message);
        }
    }

    protected File _loadFile()
    {
            File found = new File( _fileName );
            if ( !found.exists() ) {
                String message = "Failed loading file at '"+_fileName+"' of type '"+extension()+"'!\n" +
                        "It seems like the file does exist.";
                _LOGGER.error( message );
                throw new IllegalArgumentException(message);
            }
            return found;
    }


    @Override
    public FinalType free() {
        boolean success = new File(_fileName).delete();
        if ( !success ) {
            String message = "Freeing "+extension()+" file '"+_fileName+"' failed!";
            _LOGGER.error( message );
            throw new IllegalStateException( message );
        }
        return (FinalType) this;
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


}
