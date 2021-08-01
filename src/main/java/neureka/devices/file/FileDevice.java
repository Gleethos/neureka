package neureka.devices.file;


import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.devices.AbstractBaseDevice;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 *  This Device implementation is responsible for reading and or writing
 *  tensors to a given directory.
 *  The directory which ought to be governed by an instance of this
 *  class has to be passed to the constructor (as relative path) after which the device
 *  reads the files within this directory making the tensors accessible.
 *  Tensors on a file device however are not loaded onto memory entirely, instead
 *  a mere file handle for each "file tensor" is being instantiated.
 *  Therefore tensors that are stored on this device are not fit for computation.
 *  The "get(..)" method has to be called instead.
 *
 *  The abstraction provided by the "Device" interface
 *  does not necessitate that concrete implementations
 *  represent accelerator hardware. <br>
 *  Generally speaking a device is a thing that
 *  stores tensors and optionally also handles "ExecutionCall" instances.
 *  Therefore, an implementation might also represent a simple
 *  storage device like your local SSD ord HDD...
*/
public class FileDevice extends AbstractBaseDevice<Object>
{
    private static final Logger _LOG = LoggerFactory.getLogger(FileDevice.class);

    private static final Map<String, FileDevice> _DEVICES = new WeakHashMap<>();

    private Map<Tsr<Object>, FileHead<?, Object>> _stored = new HashMap<>();

    private String _directory;
    private final List<String> _loadable = new ArrayList<>();
    private final List<String> _loaded = new ArrayList<>();

    /**
     * @param path The directory path for which the responsible {@link FileDevice} instance ought to be returned.
     * @return A {@link FileDevice} instance representing the provided directory path and all compatible files within it.
     */
    public static FileDevice at( String path ) {
        FileDevice device = _DEVICES.get( path );
        if ( device != null ) return device;
        device = new FileDevice( path );
        _DEVICES.put( path, device );
        return device;
    }

    private FileDevice( String directory ) {
        _directory = directory;
        _updateFolderView();
    }

    /**
     *  The underlying folder might change, files might be added or removed.
     *  In order to have an up-to-date view of the folder this method updates the current view state.
     */
    private void _updateFolderView() {
        _loadable.clear();
        File dir = new File( _directory );
        if ( ! dir.exists() ) dir.mkdirs();
        else {
            List<String> found = new ArrayList<>();
            File[] files = dir.listFiles();
            if ( files != null ) {
                for ( File file : files ) {
                    int i = file.getName().lastIndexOf( '.' );
                    if ( i > 0 ) {
                        String extension = file.getName().substring( i + 1 );
                        if ( FileHead.FACTORY.hasLoader( extension ) ) found.add( file.getName() );
                    }
                }
                _loadable.addAll( found ); // TODO! -> Update so that new files will be detected...
            }
        }
        _loadable.removeAll(_loaded);
        _loaded.forEach( fileName -> {
              if ( !_loadable.contains(fileName) ) {
                  String message = "Missing file detected! File with name '"+fileName+"' no longer present in directory '"+_directory+"'.";
                  _LOG.warn(message);
              }
        });
    }

    public <V> Tsr<V> load( String filename ) throws IOException { return load( filename, null ); }

    public <V> Tsr<V> load( String filename, Map<String, Object> conf ) throws IOException {
        _updateFolderView();
        if ( _loadable.contains( filename ) ) {
            String extension = filename.substring( filename.lastIndexOf( '.' ) + 1 );
            FileHead<?,Object> head = FileHead.FACTORY.getLoader( extension ).load( _directory + "/" + filename, conf );
            assert head != null;
            Tsr<Object> tensor = head.load();
            _stored.put( tensor, head );
            _loadable.remove( filename );
            _loaded.add( filename );
            return (Tsr<V>) tensor;
        }
        return null;
    }

    public FileHead<?, ?> fileHeadOf( Tsr<?> tensor ) {
        return _stored.get( tensor );
    }

    @Override
    public void dispose() {
            _stored = null;
            _directory = null;
    }

    @Override
    public Device<Object> restore( Tsr<Object> tensor ) {
        if ( !this.has( tensor ) )
            throw new IllegalStateException( "The given tensor is not stored on this file device." );
        FileHead<?, Object> head = _stored.get( tensor );
        try {
            head.restore( tensor );
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public <T extends Object> Device<Object> store( Tsr<T> tensor )
    {
        if ( this.has( tensor ) ) {
            FileHead<?, Object> head = _stored.get( tensor );
            try {
                head.store( tensor );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
            return this;
        }
        String filename = tensor.shape().stream().map( Object::toString ).collect(Collectors.joining("x"));
        filename = "tensor_" + filename + "_" + tensor.getDataType().getTypeClass().getSimpleName().toLowerCase();
        filename = filename + "_" + java.time.LocalDate.now().toString();
        filename = filename + "_" + java.time.LocalTime.now().toString();
        filename = filename.replace( ".", "_" ).replace( ":","-" ) + "_.idx";
        store( tensor, filename );
        return this;
    }

    public <T extends Object> FileDevice store( Tsr<T> tensor, String filename )
    {
        return store( tensor, filename, null );
    }

    public <T extends Object> FileDevice store( Tsr<T> tensor, String filename, Map<String, Object> configurations )
    {
        String fullFileName;
        String extension;
        int i = filename.lastIndexOf( '.' );
        if ( i < 1 ) {
            fullFileName = filename + ".idx";
            extension = "idx";
        }
        else {
            extension = filename.substring( i + 1 );
            fullFileName = filename;
        }
        if ( FileHead.FACTORY.hasSaver( extension ) ) {
            _stored.put(
                    (Tsr<Object>) tensor,
                    FileHead.FACTORY.getSaver(extension).save( _directory + "/" + fullFileName, tensor, configurations )
            );
            tensor.setIsOutsourced(true);
        }
        return this;
    }

    @Override
    public <T extends Object> Device<Object> store( Tsr<T> tensor, Tsr<T> parent ) { throw new NotImplementedException(); }

    @Override
    public <T extends Object> boolean has( Tsr<T> tensor ) {
        return _stored.containsKey( tensor );
    }

    @Override
    public <T extends Object> Device<Object> free( Tsr<T> tensor )
    {
        if ( !this.has( tensor ) )
            throw new IllegalStateException( "The given tensor is not stored on this file device." );
        FileHead<?,Object> head = _stored.get( tensor );
        try {
            head.free();
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        _stored.remove( tensor );
        return this;
    }

    @Override
    public Device<Object> cleaning( Tsr<Object> tensor, Runnable action ) {
        return this;
    }

    @Override
    public Device<Object> overwrite64( Tsr<Object> tensor, double[] value ) {
        return null;
    }

    @Override
    public Device<Object> overwrite32( Tsr<Object> tensor, float[] value ) {
        return null;
    }

    @Override
    public Device<Object> swap( Tsr<Object> former, Tsr<Object> replacement ) {
        return null;
    }

    @Override
    public Device<Object> execute( ExecutionCall<Device<?>> call ) {
        throw new IllegalAccessError("FileDevice instances do not support executions.");
    }

    @Override
    public Object valueFor( Tsr<Object> tensor ) {
        return tensor.getValue();
    }

    @Override
    public Object valueFor( Tsr<Object> tensor, int index ) {
        return tensor.getValueAt( index );
    }

    @Override
    public Collection<Tsr<Object>> getTensors() {
        return _stored.keySet();
    }

    @Override
    public Operation optimizedOperationOf( Function function, String name ) {
        throw new NotImplementedException();
    }

    @Override
    public boolean update( OwnerChangeRequest<Tsr<Object>> changeRequest ) {
        Tsr<Object> oldOwner = changeRequest.getOldOwner();
        Tsr<Object> newOwner = changeRequest.getNewOwner();
        if ( _stored.containsKey( oldOwner ) ) {
            FileHead<?, Object> head = _stored.get( oldOwner );
            _stored.remove( oldOwner );
            _stored.put( newOwner, head );
        }
        changeRequest.executeChange();
        return true;
    }


    public String toString() {
        return "FileDevice(directory=" + this._directory + ", stored=" + this._stored + ", loadable=" + this._loadable + ", loaded=" + this._loaded + ")";
    }

    public String getDirectory() {
        return this._directory;
    }

    public List<String> getLoadable() {
        return this._loadable;
    }

    public List<String> getLoaded() {
        return this._loaded;
    }
}
