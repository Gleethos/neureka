package neureka.devices.file;


import neureka.DataArray;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.common.utility.LogUtil;
import neureka.devices.AbstractBaseDevice;
import neureka.devices.Device;
import neureka.common.utility.Cache;
import neureka.devices.file.handles.util.HandleFactory;
import neureka.dtype.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 *  The {@link FileDevice} is a {@link Device} implementation
 *  responsible for reading tensors from and or writing them to a given directory. <br><br>
 *
 *  The abstraction provided by the "{@link Device}" interface
 *  does not necessitate that concrete implementations
 *  represent accelerator hardware. <br>
 *  Generally speaking a device is a thing that stores tensors and optionally
 *  also expose the {@link neureka.devices.Device.Access} API for
 *  data access as well as an API useful for implementing operations...
 *  But, an implementation might also represent a simple
 *  storage device like your local SSD ord HDD, or in this case, a directory...  <br><br>
 *
 *  The directory which ought to be governed by an instance of this
 *  class has to be passed to the {@link #at(String)} factory method (as relative path),
 *  after which the files within this directory will be read, making potential tensors accessible.
 *  Tensors on a file device however are not loaded onto memory entirely, instead
 *  a mere file handle for each "file tensor" is being instantiated.
 *  Therefore, tensors that are stored on this device are not fit for computation.
 *  The {@link #restore(Tsr)} method has to be called in order to load the provided
 *  tensor back into RAM. <br><br>
 *
 *  A {@link FileDevice} can load PNG, JPG and IDX files. By default, tensors will
 *  be stored as IDX files if not explicitly specified otherwise. <br><br>
 *
*/
public final class FileDevice extends AbstractBaseDevice<Object>
{
    private static final Logger _LOG = LoggerFactory.getLogger(FileDevice.class);

    private static final Cache<Cache.LazyEntry<String, FileDevice>> _CACHE = new Cache<>(64);


    private final String _directory;
    private final List<String> _loadable = new ArrayList<>();
    private final List<String> _loaded = new ArrayList<>();
    private final Map<Tsr<Object>, FileHandle<?, Object>> _stored = new HashMap<>();


    /**
     * @param path The directory path for which the responsible {@link FileDevice} instance ought to be returned.
     * @return A {@link FileDevice} instance representing the provided directory path and all compatible files within it.
     */
    public static FileDevice at( String path ) {
        LogUtil.nullArgCheck( path, "path", String.class );
        return _CACHE.process( new Cache.LazyEntry<>( path, FileDevice::new ) ).getValue();
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
                        if ( FileHandle.FACTORY.hasLoader( extension ) ) found.add( file.getName() );
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
        LogUtil.nullArgCheck(filename, "filename", String.class);
        _updateFolderView();
        if ( _loadable.contains( filename ) ) {
            String extension = filename.substring( filename.lastIndexOf( '.' ) + 1 );
            String filePath = _directory + "/" + filename;
            HandleFactory.Loader handleLoader = FileHandle.FACTORY.getLoader( extension );
            if ( handleLoader == null )
                throw new IllegalStateException(
                    "Failed to create file handle loader for file with extension '" + extension + "'."
                );

            FileHandle<?,Object> handle = handleLoader.load( filePath, conf );
            if ( handle == null )
                throw new IllegalStateException(
                    "Failed to create file handle for file path '" + filePath + " and loading conf '" + conf + "'."
                );

            Tsr<Object> tensor = handle.load();
            _stored.put( tensor, handle );
            _loadable.remove( filename );
            _loaded.add( filename );
            return (Tsr<V>) tensor;
        }
        return null;
    }

    public FileHandle<?, ?> fileHandleOf( Tsr<?> tensor ) {
        LogUtil.nullArgCheck(tensor, "tensor", Tsr.class);
        return _stored.get( tensor );
    }

    @Override
    public void dispose() {
        _stored.clear();
        _loadable.clear();
        _loaded.clear();
    }

    /** {@inheritDoc} */
    @Override
    public Device<Object> restore( Tsr<Object> tensor ) {
        LogUtil.nullArgCheck(tensor, "tensor", Tsr.class);
        if ( !this.has( tensor ) )
            throw new IllegalStateException( "The given tensor is not stored on this file device." );
        FileHandle<?, Object> head = _stored.get( tensor );
        try {
            head.restore( tensor );
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public <T> Device<Object> store( Tsr<T> tensor ) {
        LogUtil.nullArgCheck(tensor, "tensor", Tsr.class);
        if ( this.has( tensor ) ) {
            FileHandle<?, Object> head = _stored.get( tensor );
            try {
                head.store( tensor );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
            return this;
        }
        String filename = tensor.shape().stream().map( Object::toString ).collect(Collectors.joining("x"));
        filename = "tensor_" + filename + "_" + tensor.getDataType().getRepresentativeType().getSimpleName().toLowerCase();
        filename = filename + "_" + java.time.LocalDate.now();
        filename = filename + "_" + java.time.LocalTime.now().toString();
        filename = filename.replace( ".", "_" ).replace( ":","-" ) + "_.idx";
        store( tensor, filename );
        return this;
    }

    public <T> FileDevice store( Tsr<T> tensor, String filename ) {
        return this.store( tensor, filename, null );
    }

    public <T> FileDevice store( Tsr<T> tensor, String filename, Map<String, Object> configurations ) {
        LogUtil.nullArgCheck(tensor, "tensor", Tsr.class);
        LogUtil.nullArgCheck( filename, "filename", String.class );
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
        if ( FileHandle.FACTORY.hasSaver( extension ) ) {
            _stored.put(
                    (Tsr<Object>) tensor,
                    FileHandle.FACTORY.getSaver(extension).save( _directory + "/" + fullFileName, tensor, configurations )
            );
            tensor.getUnsafe().setDataArray(null);
        }
        return this;
    }

    @Override
    public <T> Device<Object> store( Tsr<T> tensor, Tsr<T> parent ) { throw new IllegalStateException(); }

    @Override
    public <T> boolean has( Tsr<T> tensor ) {
        LogUtil.nullArgCheck(tensor, "tensor", Tsr.class);
        return _stored.containsKey( tensor );
    }

    @Override
    public <T> Device<Object> free( Tsr<T> tensor ) {
        LogUtil.nullArgCheck(tensor, "tensor", Tsr.class);
        if ( !this.has( tensor ) )
            throw new IllegalStateException( "The given tensor is not stored on this file device." );
        FileHandle<?,Object> head = _stored.get( tensor );
        try {
            head.free();
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        _stored.remove( tensor );
        return this;
    }

    @Override
    public <T> Access<T> access( Tsr<T> tensor) {
        throw new IllegalAccessError(
                this.getClass().getSimpleName()+" instances do not support accessing the state of a stored tensor."
            );
    }

    @Override
    public Device<Object> approve( ExecutionCall<? extends Device<?>> call ) {
        throw new IllegalAccessError(
                this.getClass().getSimpleName()+" instances do not support executions on stored tensors."
            );
    }

    @Override
    public Collection<Tsr<Object>> getTensors() { return _stored.keySet(); }

    @Override
    public DataArray allocate(DataType<?> dataType, int size) {
        throw new IllegalStateException("FileDevice instances do not support allocation of memory.");
    }

    @Override
    public <V> DataArray allocate(DataType<V> dataType, int size, V initialValue) {
        throw new IllegalStateException("FileDevice instances do not support allocation of memory.");
    }

    @Override
    public DataArray allocate(Object jvmData, int desiredSize) {
        throw new IllegalStateException("FileDevice instances do not support allocation of memory.");
    }

    @Override
    public Operation optimizedOperationOf( Function function, String name ) {
        throw new IllegalStateException(
                this.getClass().getSimpleName()+" instances do not support operations!"
            );
    }

    @Override
    public boolean update( OwnerChangeRequest<Tsr<Object>> changeRequest ) {
        Tsr<Object> oldOwner = changeRequest.getOldOwner();
        Tsr<Object> newOwner = changeRequest.getNewOwner();
        if ( _stored.containsKey( oldOwner ) ) {
            FileHandle<?, Object> head = _stored.get( oldOwner );
            _stored.remove( oldOwner );
            _stored.put( newOwner, head );
        }
        changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
        return true;
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName()+"[directory=" + _directory + ",stored=" + _stored + ",loadable=" + _loadable + ",loaded=" + _loaded + "]";
    }

    public String getDirectory() { return _directory; }

    public List<String> getLoadable() { return new ArrayList<>(_loadable); }

    public List<String> getLoaded() { return new ArrayList<>(_loaded); }

}
