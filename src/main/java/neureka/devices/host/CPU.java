package neureka.devices.host;

import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.devices.AbstractDevice;
import neureka.devices.Device;
import neureka.devices.host.concurrent.Parallelism;
import neureka.devices.host.concurrent.WorkScheduler;
import neureka.devices.host.machine.ConcreteMachine;
import neureka.dtype.DataType;
import neureka.dtype.custom.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntSupplier;

/**
 *  The CPU class, one of many implementations of the {@link Device} interface,
 *  is simply supposed to be an API for dispatching threaded workloads onto the CPU
 *  as well as reading from or writing to tensors it stores.
 *  Contrary to other types of devices, the CPU will represent a tensors' data by default, simply
 *  because the tensors will be stored in RAM (JVM heap) by default if no device was specified.
 *  This means that they are implicitly "stored" on the {@link CPU} device.
 *  The class is also a singleton instead of being part of a {@link neureka.backend.api.BackendExtension}.
 */
public class CPU extends AbstractDevice<Object>
{
    private static final Logger _LOG = LoggerFactory.getLogger( CPU.class );
    private static final CPU _INSTANCE;

    private static final WorkScheduler.Divider _DIVIDER;
    private static final IntSupplier _PARALLELISM;

    public static final int PARALLELIZATION_THRESHOLD = 32;
    public static final String THREAD_PREFIX = "neureka-daemon";

    static {
        _INSTANCE = new CPU();
        _DIVIDER = new WorkScheduler.Divider(_INSTANCE._executor._pool);
        _PARALLELISM = Parallelism.THREADS;
    }

    private final JVMExecutor _executor = new JVMExecutor();
    private final Set<Tsr<Object>> _tensors = Collections.newSetFromMap(new WeakHashMap<>());

    private CPU() { super(); }

    /**
     *  Use this method to access the singleton instance of this {@link CPU} class,
     *  which is a {@link Device} type and default location for freshly instantiated {@link Tsr} instances.
     *  {@link Tsr} instances located on the {@link CPU} device will reside in regular RAM
     *  causing operations to run on the JVM and thereby the CPU.
     *
     * @return The singleton instance of this {@link CPU} class.
     */
    public static CPU get() { return _INSTANCE; }

    /**
     *  The {@link JVMExecutor} offers a similar functionality as the parallel stream API,
     *  however it differs in that the {@link JVMExecutor} is processing {@link RangeWorkload} lambdas
     *  instead of simply exposing a single index or concrete elements for a given workload size.
     *
     * @return A parallel range based execution API running on the JVM.
     */
    public JVMExecutor getExecutor() { return _executor; }

    @Override
    protected boolean _approveExecutionOf( Tsr<?>[] tensors, int d, Operation operation ) { return true; }

    /**
     *  This method will shut down the internal thread-pool used by this
     *  class to execute JVM/CPU based operations in parallel.
     */
    @Override
    public void dispose() {
        _executor._pool.shutdown();
        _tensors.clear();
        _LOG.warn(
            "Main thread pool in '"+this.getClass()+"' shutting down! " +
            "Newly incoming operations will not be executed in parallel."
        );
    }

    @Override
    public CPU restore( Tsr<Object> tensor ) { return this; }

    @Override
    public <T> CPU store( Tsr<T> tensor ) {
        if ( !this.has( tensor ) )
            tensor.getUnsafe().getData().owner().restore( tensor );

        _tensors.add( (Tsr<Object>) tensor);
        return this;
    }

    @Override protected final <T> void _updateNDConf(Tsr<T> tensor) { /* Nothing to do here */ }

    @Override
    protected final <T> int _sizeOccupiedBy(Tsr<T> tensor) {
        Object data = tensor.getUnsafe().getData().getRef();
        if      ( data instanceof float[]   ) return ( (float[])   data).length;
        else if ( data instanceof double[]  ) return ( (double[])  data).length;
        else if ( data instanceof short[]   ) return ( (short[])   data).length;
        else if ( data instanceof int[]     ) return ( (int[])     data).length;
        else if ( data instanceof byte[]    ) return ( (byte[])    data).length;
        else if ( data instanceof long[]    ) return ( (long[])    data).length;
        else if ( data instanceof boolean[] ) return ( (boolean[]) data).length;
        else if ( data instanceof char[]    ) return ( (char[])    data).length;
        else return ( (Object[]) data).length;
    }

    @Override
    protected final <T> Object _readAll(Tsr<T> tensor, boolean clone ) {
        Object data = tensor.getUnsafe().getData().getRef();
        if ( clone ) {
            if ( data instanceof double[]  ) return ( (double[])  data ).clone();
            if ( data instanceof float[]   ) return ( (float[])   data ).clone();
            if ( data instanceof byte[]    ) return ( (byte[])    data ).clone();
            if ( data instanceof short[]   ) return ( (short[])   data ).clone();
            if ( data instanceof int[]     ) return ( (int[])     data ).clone();
            if ( data instanceof long[]    ) return ( (long[])    data ).clone();
            if ( data instanceof char[]    ) return ( (char[])    data ).clone();
            if ( data instanceof boolean[] ) return ( (boolean[]) data ).clone();
            if ( data instanceof Object[]  ) return ( (Object[])  data ).clone();
        }
        return data;
    }

    @Override
    protected final <T> T _readItem( Tsr<T> tensor, int index ) {
        Object data = tensor.getUnsafe().getData().getRef();
        if      ( data instanceof float[]   ) return (T)Float.valueOf( ((float[])   data)[ index ] );
        else if ( data instanceof double[]  ) return (T)Double.valueOf( ((double[])  data)[ index ] );
        else if ( data instanceof short[]   ) return (T)Short.valueOf( ((short[])   data)[ index ] );
        else if ( data instanceof int[]     ) return (T)Integer.valueOf( ((int[])     data)[ index ] );
        else if ( data instanceof byte[]    ) return (T)Byte.valueOf( ((byte[])    data)[ index ] );
        else if ( data instanceof long[]    ) return (T)Long.valueOf( ((long[])    data)[ index ] );
        else if ( data instanceof boolean[] ) return (T)Boolean.valueOf( ((boolean[]) data)[ index ] );
        else if ( data instanceof char[]    ) return (T)Character.valueOf( ((char[])  data)[ index ] );
        else return (T)( (Object[]) data)[ index ];
    }

    @Override
    protected final <T, A> A _readArray(
            Tsr<T> tensor, Class<A> arrayType, int start, int size
    ) {
        if ( arrayType == float[].class ) {
            float[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), float[].class);
            float[] data = new float[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == short[].class ){
            short[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), short[].class);
            short[] data = new short[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == byte[].class ){
            byte[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), byte[].class);
            byte[] data = new byte[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == boolean[].class ){
            boolean[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), boolean[].class);
            boolean[] data = new boolean[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == char[].class ){
            char[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), char[].class);
            char[] data = new char[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == double[].class ){
            double[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), double[].class);
            return (A) java.util.Arrays.stream(source, start, start + size).toArray();
        } else if ( arrayType == int[].class ){
            int[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), int[].class);
            return (A) java.util.Arrays.stream(source, start, start + size).toArray();
        } else if ( arrayType == long[].class ){
            long[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), long[].class);
            return (A) java.util.Arrays.stream(source, start, start + size).toArray();
        } else if ( arrayType == Object[].class ){
            Object[] source = DataConverter.get().convert(tensor.getUnsafe().getData().getRef(), Object[].class);
            return (A) java.util.Arrays.stream(source, start, start + size).toArray();
        }
        throw new IllegalArgumentException("Array type '"+arrayType.getSimpleName()+"' not supported!");
    }

    @Override
    protected final <T> void _writeItem( Tsr<T> tensor, T item, int start, int size ) {
        Object data = tensor.getUnsafe().getData().getRef();
        Class<?> arrayType = data.getClass();
        if ( arrayType == float[].class ) {
            float source = DataConverter.get().convert(item, Float.class);
            float[] target = (float[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = source;
        } else if ( arrayType == short[].class ){
            short source = DataConverter.get().convert(item, Short.class);
            short[] target = (short[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = source;
        } else if ( arrayType == byte[].class ){
            byte source = DataConverter.get().convert(item, Byte.class);
            byte[] target = (byte[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = source;
        } else if ( arrayType == boolean[].class ){
            boolean source = DataConverter.get().convert(item, Boolean.class);
            boolean[] target = (boolean[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = source;
        } else if ( arrayType == double[].class ){
            double source = DataConverter.get().convert(item, Double.class);
            double[] target = (double[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = source;
        } else if ( arrayType == int[].class ){
            int source = DataConverter.get().convert(item, Integer.class);
            int[] target = (int[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = source;
        } else if ( arrayType == long[].class ){
            long source = DataConverter.get().convert(item, Long.class);
            long[] target = (long[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = source;
        } else if ( arrayType == char[].class ){
            char source = DataConverter.get().convert(item, Character.class);
            char[] target = (char[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = source;
        } else if ( arrayType == Object[].class ) {
            Object[] target = (Object[]) data;
            for ( int i = start; i < (start+size); i++ ) target[i] = item;
        }
    }

    @Override
    protected final <T> void _writeArray(
            Tsr<T> tensor, Object array, int offset, int start, int size
    ) {
        Object data = tensor.getUnsafe().getData() == null ? null : tensor.getUnsafe().getData().getRef();
        if ( data == null ) {
            tensor.getUnsafe().setData( _dataArrayOf(array) );
            return;
        }
        Class<?> arrayType = data.getClass();
        if ( arrayType == float[].class ) {
            float[] source = DataConverter.get().convert(array, float[].class);
            float[] target = (float[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        } else if ( arrayType == short[].class ){
            short[] source = DataConverter.get().convert(array, short[].class);
            short[] target = (short[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        } else if ( arrayType == byte[].class ){
            byte[] source = DataConverter.get().convert(array, byte[].class);
            byte[] target = (byte[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        } else if ( arrayType == boolean[].class ){
            boolean[] source = DataConverter.get().convert(array, boolean[].class);
            boolean[] target = (boolean[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        } else if ( arrayType == double[].class ){
            double[] source = DataConverter.get().convert(array, double[].class);
            double[] target = (double[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        } else if ( arrayType == int[].class ){
            int[] source = DataConverter.get().convert(array, int[].class);
            int[] target = (int[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        } else if ( arrayType == char[].class ){
            char[] source = DataConverter.get().convert(array, char[].class);
            char[] target = (char[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        } else if ( arrayType == long[].class ){
            long[] source = DataConverter.get().convert(array, long[].class);
            long[] target = (long[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        } else if ( arrayType == Object[].class ){
            Object[] source = DataConverter.get().convert(array, Object[].class);
            Object[] target = (Object[]) data;
            System.arraycopy(source, offset, target, start, Math.min(size, source.length));
        }
        else throw new IllegalArgumentException("Array type '"+arrayType.getSimpleName()+"' not supported!");
    }

    @Override
    public final neureka.Data allocate( DataType<?> dataType, int size ) {
        Class<?> typeClass = dataType.getRepresentativeType();
        if ( typeClass == F64.class )
            return _dataArrayOf(new double[ size ]);
        else if ( typeClass == F32.class )
            return _dataArrayOf(new float[ size ]);
        else if ( typeClass == I32.class || typeClass == UI32.class )
            return _dataArrayOf(new int[ size ]);
        else if ( typeClass == I16.class || typeClass == UI16.class )
            return _dataArrayOf(new short[ size ]);
        else if ( typeClass == I8.class || typeClass == UI8.class )
            return _dataArrayOf(new byte[ size ]);
        else if ( typeClass == I64.class || typeClass == UI64.class )
            return _dataArrayOf(new long[ size ]);
        else if ( dataType.getItemTypeClass() == Boolean.class )
            return _dataArrayOf(new boolean[ size ]);
        else if ( dataType.getItemTypeClass() == Character.class )
            return _dataArrayOf(new char[ size ]);
        else
            return _dataArrayOf(new Object[ size ]);
    }

    @Override
    public <V> neureka.Data allocate( DataType<V> dataType, int size, V initialValue ) {
        Class<?> type = dataType.getItemTypeClass();
        neureka.Data array = allocate( dataType, size );
        Object data = array.getRef();
        if      ( type == Double   .class ) { Arrays.fill((double[])  data, (Double)   initialValue); }
        else if ( type == Float    .class ) { Arrays.fill((float[])   data, (Float)    initialValue); }
        else if ( type == Integer  .class ) { Arrays.fill((int[])     data, (Integer)  initialValue); }
        else if ( type == Short    .class ) { Arrays.fill((short[])   data, (Short)    initialValue); }
        else if ( type == Byte     .class ) { Arrays.fill((byte[])    data, (Byte)     initialValue); }
        else if ( type == Long     .class ) { Arrays.fill((long[])    data, (Long)     initialValue); }
        else if ( type == Boolean  .class ) { Arrays.fill((boolean[]) data, (Boolean)  initialValue); }
        else if ( type == Character.class ) { Arrays.fill((char[])    data, (Character)initialValue); }
        else { Arrays.fill((Object[])  data, initialValue); }
        return array;
    }

    @Override
    public neureka.Data allocate(Object jvmData, int desiredSize )
    {
        neureka.Data data = _dataArrayOf(jvmData);
        if ( jvmData instanceof int[] ) {
            int[] array = (int[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate( DataType.of(I32.class), desiredSize );
                for ( int i = 0; i < desiredSize; i++ ) ( (int[]) data.getRef() )[ i ]  = array[ i % array.length ];
            }
            return data;
        } else if ( jvmData instanceof float[] ) {
            float[] array = (float[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate( DataType.of(F32.class), desiredSize );
                for ( int i = 0; i < desiredSize; i++ ) ( (float[]) data.getRef() )[ i ]  = array[ i % array.length ];
            }
            return data;
        } else if ( jvmData instanceof double[] ) {
            double[] array = (double[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate( DataType.of(F64.class), desiredSize );
                for ( int i = 0; i < desiredSize; i++ ) ( (double[]) data.getRef() )[ i ]  = array[ i % array.length ];
            }
            return data;
        } else if ( jvmData instanceof long[] ) {
            long[] array = (long[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate( DataType.of(I64.class), desiredSize );
                for ( int i = 0; i < desiredSize; i++ ) ( (long[]) data.getRef() )[ i ]  = array[ i % array.length ];
            }
            return data;
        } else if ( jvmData instanceof short[] ) {
            short[] array = (short[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate( DataType.of(I16.class), desiredSize );
                for ( int i = 0; i < desiredSize; i++ ) ( (short[]) data.getRef() )[ i ]  = array[ i % array.length ];
            }
            return data;
        } else if ( jvmData instanceof byte[] ) {
            byte[] array = (byte[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate(DataType.of(I8.class), desiredSize);
                for (int i = 0; i < desiredSize; i++) ((byte[]) data.getRef())[i] = array[i % array.length];
            }
            return data;
        } else if ( jvmData instanceof boolean[] ) {
            boolean[] array = (boolean[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate(DataType.of(Boolean.class), desiredSize);
                for (int i = 0; i < desiredSize; i++) ((boolean[]) data.getRef())[i] = array[i % array.length];
            }
            return data;
        } else if ( jvmData instanceof char[] ) {
            char[] array = (char[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate(DataType.of(Character.class), desiredSize);
                for (int i = 0; i < desiredSize; i++) ((char[]) data.getRef())[i] = array[i % array.length];
            }
            return data;
        } else if ( jvmData instanceof Object[] ) {
            Object[] array = (Object[]) jvmData;
            if ( desiredSize != array.length ) {
                data = CPU.get().allocate(DataType.of(Object.class), desiredSize);
                for (int i = 0; i < desiredSize; i++) ((Object[]) data.getRef())[i] = array[i % array.length];
            }
            return data;
        }
        else
            throw new IllegalArgumentException("Array type '"+jvmData.getClass().getSimpleName()+"' not supported!");
    }

    public neureka.Data allocate( Object data ) {
        int size;
        if ( data instanceof Object[] ) {
            size = ( (Object[]) data ).length;
        } else if ( data instanceof int[] ) {
            size = ( (int[]) data ).length;
        } else if ( data instanceof long[] ) {
            size = ( (long[]) data ).length;
        } else if ( data instanceof float[] ) {
            size = ( (float[]) data ).length;
        } else if ( data instanceof double[] ) {
            size = ( (double[]) data ).length;
        } else if ( data instanceof short[] ) {
            size = ( (short[]) data ).length;
        } else if ( data instanceof byte[] ) {
            size = ( (byte[]) data ).length;
        } else if ( data instanceof boolean[] ) {
            size = ( (boolean[]) data ).length;
        } else if ( data instanceof char[] ) {
            size = ( (char[]) data ).length;
        }
        else
            throw new IllegalArgumentException( "Unsupported data type: " + data.getClass() );

        neureka.Data dataArray = CPU.get().allocate( data, size );
        if ( dataArray.getRef() != data )
            throw new IllegalStateException( "CPU seems to have reallocated some already valid data unnecessarily! This is most likely a bug." );

        return dataArray;
    }

    @Override
    protected final neureka.Data<Object> _actualize(Tsr<?> tensor ) {
        neureka.Data<Object> data = (neureka.Data<Object>) tensor.getUnsafe().getData();
        Object value = data.getRef();
        DataType<?> dataType = tensor.getDataType();
        int size = tensor.size();
        Class<?> typeClass = dataType.getRepresentativeType();
        Object newValue;
        if ( typeClass == F64.class ) {
            if ( ( (double[]) value ).length == size ) return data;
            newValue = new double[ size ];
            if ( ( (double[]) value )[ 0 ] != 0d ) Arrays.fill( (double[]) newValue, ( (double[]) value )[ 0 ] );
        } else if ( typeClass == F32.class ) {
            if ( ( (float[]) value ).length == size ) return data;
            newValue = new float[size];
            if ( ( (float[]) value )[ 0 ] != 0f ) Arrays.fill( (float[]) newValue, ( (float[]) value )[ 0 ] );
        } else if ( typeClass == I32.class ) {
            if ( ( (int[]) value ).length == size ) return data;
            newValue = new int[ size ];
            if ( ( (int[]) value )[ 0 ] != 0 ) Arrays.fill( (int[]) newValue, ( (int[]) value )[ 0 ] );
        } else if ( typeClass == I16.class ) {
            if ( ( (short[]) value ).length == size ) return data;
            newValue = new short[ size ];
            if ( ( (short[]) value )[ 0 ] != 0 ) Arrays.fill( (short[]) newValue, ( (short[]) value )[ 0 ] );
        } else if ( typeClass == I8.class ) {
            if ( ( (byte[]) value ).length == size ) return data;
            newValue = new byte[ size ];
            if ( ( (byte[]) value )[ 0 ] != 0 ) Arrays.fill( (byte[]) newValue, ( (byte[]) value )[ 0 ] );
        } else if ( typeClass == I64.class ) {
            if ( ( (long[]) value ).length == size ) return data;
            newValue = new long[ size ];
            if ( ( (long[]) value )[ 0 ] != 0 ) Arrays.fill( (long[]) newValue, ( (long[]) value )[ 0 ] );
        } else if ( typeClass == Boolean.class ) {
            if ( ( (boolean[]) value ).length == size ) return data;
            newValue = new boolean[ size ];
            Arrays.fill( (boolean[]) newValue, ( (boolean[]) value )[ 0 ] );
        } else if ( typeClass == Character.class ) {
            if ( ( (char[]) value ).length == size ) return data;
            newValue = new char[ size ];
            if ( ( (char[]) value )[ 0 ] != (char) 0 ) Arrays.fill( (char[]) newValue, ( (char[]) value )[ 0 ] );
        } else {
            if ( ( (Object[]) value ).length == size ) return data;
            newValue = new Object[ size ];
            if ( ( (Object[]) value )[ 0 ] != null ) Arrays.fill( (Object[]) newValue, ( (Object[]) value )[ 0 ] );
        }
        return _dataArrayOf(newValue);
    }

    @Override
    public <T> CPU free( Tsr<T> tensor ) {
        LogUtil.nullArgCheck( tensor, "tensor", Tsr.class );
        _tensors.remove( tensor );
        return this;
    }

    @Override
    protected <T> void _swap( Tsr<T> former, Tsr<T> replacement ) {}

    @Override
    public Collection<Tsr<Object>> getTensors() { return _tensors; }

    @Override
    public Operation optimizedOperationOf( Function function, String name ) { throw new IllegalStateException(); }

    /**
     *  This method is part of the component system built into the {@link Tsr} class.
     *  Do not use this as part of anything but said component system.
     *
     * @param changeRequest An API which describes the type of update and a method for executing said update.
     * @return The truth value determining if this {@link Device} ought to be added to a tensor (Here always false!).
     */
    @Override
    public boolean update( OwnerChangeRequest<Tsr<Object>> changeRequest ) {
        super.update( changeRequest );
        return false; // This type of device can not be a component simply because it is the default device
    }

    /**
     * Returns the number of CPU cores available to the Java virtual machine.
     * This value may change during a particular invocation of the virtual machine.
     * Applications that are sensitive to the number of available processors should
     * therefore occasionally poll this property and adjust their resource usage appropriately.
     *
     * @return The maximum number of CPU cores available to the JVM.
     *         This number is never smaller than one!
     */
    public int getCoreCount() { return Runtime.getRuntime().availableProcessors(); }

    @Override
    public String toString() { return this.getClass().getSimpleName()+"[cores="+getCoreCount()+"]"; }

    /**
     *  A simple functional interface for executing a range whose implementations will
     *  either be executed sequentially or they are being dispatched to
     *  a thread-pool, given that the provided workload is large enough.
     */
    @FunctionalInterface
    public interface RangeWorkload { void execute( int start, int end );  }


    @FunctionalInterface
    public interface IndexedWorkload { void execute( int i );  }

    /**
     *  The {@link JVMExecutor} offers a similar functionality as the parallel stream API,
     *  however it differs in that the {@link JVMExecutor} is processing {@link RangeWorkload} lambdas
     *  instead of simply exposing a single index or concrete elements for a given workload size.
     *  This means that a {@link RangeWorkload} lambda will be called with the work range of a single worker thread
     *  processing its current workload.
     *  This range is dependent on the number of available threads as well as the size of the workload.
     *  If the workload is very small, then the current main thread will process the entire workload range
     *  whereas the underlying {@link ThreadPoolExecutor} will not be used to avoid unnecessary overhead.
     */
    public static class JVMExecutor
    {
        private static final AtomicInteger _COUNTER = new AtomicInteger();
        private static final ThreadGroup   _GROUP   = new ThreadGroup(THREAD_PREFIX+"-group");

        /*
            The following 2 constants determine if any given workload size will be parallelized or not...
            We might want to adjust this some more for better performance...
         */
        private static final int _MIN_THREADED_WORKLOAD_SIZE = 32;
        private static final int _MIN_WORKLOAD_PER_THREAD    = 8;

        private final ThreadPoolExecutor _pool =
                                            new ThreadPoolExecutor(
                                                    ConcreteMachine.ENVIRONMENT.units,
                                                    Integer.MAX_VALUE,
                                                    5L,
                                                    TimeUnit.SECONDS,
                                                    new SynchronousQueue<Runnable>(), // This is basically always of size 1
                                                    _newThreadFactory(THREAD_PREFIX+"-")
                                            );

        private static ThreadFactory _newThreadFactory( final String name ) {
            return _newThreadFactory( _GROUP, name );
        }

        private static ThreadFactory _newThreadFactory( final ThreadGroup group, final String name ) {

            String prefix = name.endsWith("-") ? name : name + "-";

            return target -> {
                Thread thread = new Thread(
                                    group, target,
                                    prefix + _COUNTER.incrementAndGet() // The name, including the thread number.
                                );
                thread.setDaemon(true);
                return thread;
            };
        }

        /**
         * Returns the approximate number of threads that are actively
         * executing tasks.
         *
         * @return the number of threads
         */
        public int getActiveThreadCount() { return _pool.getActiveCount(); }

        /**
         * Returns the core number of threads.
         *
         * @return the core number of threads
         */
        public int getCorePoolSize() { return _pool.getCorePoolSize(); }

        /**
         * Returns the approximate total number of tasks that have
         * completed execution. Because the states of tasks and threads
         * may change dynamically during computation, the returned value
         * is only an approximation, but one that does not ever decrease
         * across successive calls.
         *
         * @return the number of tasks
         */
        public long getCompletedTaskCount() { return _pool.getCompletedTaskCount(); }

        /**
         *  This method slices the provided workload size into multiple ranges which can be executed in parallel.
         *
         * @param workloadSize The total workload size which ought to be split into multiple ranges.
         * @param workload The range lambda which ought to be executed across multiple threads.
         */
        public void threaded( int workloadSize, RangeWorkload workload )
        {
            LogUtil.nullArgCheck( workload, "workload", RangeWorkload.class );
            int cores = get().getCoreCount();
            cores = ( cores == 0 ? 1 : cores );
            if ( workloadSize >= _MIN_THREADED_WORKLOAD_SIZE && ( ( workloadSize / cores ) >= _MIN_WORKLOAD_PER_THREAD) ) {
                threaded(0, workloadSize, workload );
            }
            else sequential( workloadSize, workload );
        }

        /**
         *  Executes the provided workload lambda across multiple threads
         *  where the provided worker lambda will receive the index/id of the current worker.
         *
         * @param numberOfWorkloads The total number of workloads to be executed.
         * @param workload The workload lambda to be executed.
         */
        public void threaded( int numberOfWorkloads, IndexedWorkload workload ) {
            LogUtil.nullArgCheck( workload, "workload", IndexedWorkload.class );
            _DIVIDER.parallelism( _PARALLELISM )
                    .threshold( 1 )
                    .submit( numberOfWorkloads, (i)-> workload.execute(i) );
        }

        /**
         *  This method will simply execute the provided {@link RangeWorkload} lambda sequentially
         *  with 0 as the start index and {@code workloadSize} as the exclusive range.       <br><br>
         *
         * @param workloadSize The workload size which will be passed to the provided {@link RangeWorkload} as second argument.
         * @param workload The {@link RangeWorkload} which will be executed sequentially.
         */
        public void sequential( int workloadSize, RangeWorkload workload ) {
            LogUtil.nullArgCheck( workload, "workload", RangeWorkload.class );
            workload.execute( 0, workloadSize );
        }


        /**
         *  Takes the provided range and divides it into multithreaded workloads.
         *
         * @param first The start index of the threaded workload range.
         * @param limit The limit for the workload range, which is exclusive.
         * @param rangeWorkload A workload lambda which will be called by different threads with different sub-ranges.
         */
        public void threaded(
                final int first,
                final int limit,
                final RangeWorkload rangeWorkload
        ) {
            LogUtil.nullArgCheck( rangeWorkload, "rangeWorkload", RangeWorkload.class );
            _DIVIDER.parallelism( _PARALLELISM )
                    .threshold( PARALLELIZATION_THRESHOLD )
                    .divide( first, limit, rangeWorkload);
        }

    }

}