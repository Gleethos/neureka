package neureka.devices.host;

import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.common.utility.DataConverter;
import neureka.devices.AbstractDevice;
import neureka.devices.Device;
import neureka.devices.host.concurrent.Parallelism;
import neureka.devices.host.concurrent.WorkScheduler;
import neureka.devices.host.machine.ConcreteMachine;
import neureka.dtype.DataType;
import neureka.dtype.custom.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.*;
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
    public <T> CPU store(Tsr<T> tensor ) {
        //super.store(tensor);
        _tensors.add( (Tsr<Object>) tensor);
        return this;
    }

    @Override protected final <T> void _updateNDConf(Tsr<T> tensor) { /* Nothing to do here */ }

    @Override
    protected final <T> int _sizeOccupiedBy(Tsr<T> tensor) {
        Object data = tensor.getUnsafe().getData();
        if      ( data instanceof float[] )  return ( (float[])   data).length;
        else if ( data instanceof double[] ) return ( (double[])  data).length;
        else if ( data instanceof short[] )  return ( (short[])   data).length;
        else if ( data instanceof int[] )    return ( (int[])     data).length;
        else if ( data instanceof byte[] )   return ( (byte[])    data).length;
        else if ( data instanceof long[] )   return ( (long[])    data).length;
        else if ( data instanceof boolean[] )return ( (boolean[]) data).length;
        else if ( data instanceof char[] )   return ( (char[])    data).length;
        else return ( (Object[]) data).length;
    }

    @Override
    protected final <T> Object _readAll(Tsr<T> tensor, boolean clone ) {
        Object data = tensor.getUnsafe().getData();
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
        Object data = tensor.getUnsafe().getData();
        if      ( data instanceof float[] )  return (T)(Float)  ( (float[])   data)[ index ];
        else if ( data instanceof double[] ) return (T)(Double) ( (double[])  data)[ index ];
        else if ( data instanceof short[] )  return (T)(Short)  ( (short[])   data)[ index ];
        else if ( data instanceof int[] )    return (T)(Integer)( (int[])     data)[ index ];
        else if ( data instanceof byte[] )   return (T)(Byte)   ( (byte[])    data)[ index ];
        else if ( data instanceof long[] )   return (T)(Long)   ( (long[])    data)[ index ];
        else if ( data instanceof boolean[] )return (T)(Boolean)( (boolean[]) data)[ index ];
        else if ( data instanceof char[] )   return (T)(Character)( (char[])  data)[ index ];
        else return (T)( (Object[]) data)[ index ];
    }

    @Override
    protected final <T, A> A _readArray(
            Tsr<T> tensor, Class<A> arrayType, int start, int size
    ) {
        if ( arrayType == float[].class ) {
            float[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), float[].class);
            float[] data = new float[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == short[].class ){
            short[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), short[].class);
            short[] data = new short[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == byte[].class ){
            byte[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), byte[].class);
            byte[] data = new byte[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == boolean[].class ){
            boolean[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), boolean[].class);
            boolean[] data = new boolean[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == char[].class ){
            char[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), char[].class);
            char[] data = new char[size];
            System.arraycopy(source, start, data, 0, size);
            return (A) data;
        } else if ( arrayType == double[].class ){
            double[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), double[].class);
            return (A) java.util.Arrays.stream(source, start, start + size).toArray();
        } else if ( arrayType == int[].class ){
            int[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), int[].class);
            return (A) java.util.Arrays.stream(source, start, start + size).toArray();
        } else if ( arrayType == long[].class ){
            long[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), long[].class);
            return (A) java.util.Arrays.stream(source, start, start + size).toArray();
        } else if ( arrayType == Object[].class ){
            Object[] source = DataConverter.get().convert(tensor.getUnsafe().getData(), Object[].class);
            return (A) java.util.Arrays.stream(source, start, start + size).toArray();
        }
        throw new IllegalArgumentException("Array type '"+arrayType.getSimpleName()+"' not supported!");
    }

    @Override
    protected final <T> void _writeItem( Tsr<T> tensor, T item, int start, int size ) {
        Object data = tensor.getUnsafe().getData();
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
    protected final <T> void _writeArray(Tsr<T> tensor, Object array, int offset, int start, int size) {
        Object data = tensor.getUnsafe().getData();
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
    protected Object _allocate( DataType<?> dataType, int size ) {
        Class<?> typeClass = dataType.getRepresentativeType();
        if ( typeClass == F64.class )
            return new double[ size ];
        else if ( typeClass == F32.class )
            return new float[ size ];
        else if ( typeClass == I32.class || typeClass == UI32.class )
            return new int[ size ];
        else if ( typeClass == I16.class || typeClass == UI16.class )
            return new short[ size ];
        else if ( typeClass == I8.class || typeClass == UI8.class )
            return new byte[ size ];
        else if ( typeClass == I64.class || typeClass == UI64.class )
            return new long[ size ];
        else if ( dataType.getValueTypeClass() == Boolean.class )
            return new boolean[ size ];
        else if ( dataType.getValueTypeClass() == Character.class )
            return new char[ size ];
        else
            return new Object[ size ];
    }

    @Override
    public <T> CPU store(Tsr<T> tensor, Tsr<T> parent ) {
        _tensors.add( (Tsr<Object>) tensor);
        _tensors.add( (Tsr<Object>) parent);
        return this;
    }

    @Override
    public <T> boolean has(Tsr<T> tensor ) { return _tensors.contains( tensor ); }

    @Override
    public <T> CPU free(Tsr<T> tensor ) {
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
    public String toString() { return this.getClass().getSimpleName()+"[coreCount="+getCoreCount()+"]"; }

    /**
     *  A simple functional interface for executing a range whose implementations will
     *  either be executed sequentially or they are being dispatched to
     *  a thread-pool, given that the provided workload is large enough.
     */
    @FunctionalInterface
    public interface RangeWorkload {  void execute( int start, int end );  }

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
            int cores = get().getCoreCount();
            cores = ( cores == 0 ? 1 : cores );
            if ( workloadSize >= _MIN_THREADED_WORKLOAD_SIZE && ( ( workloadSize / cores ) >= _MIN_WORKLOAD_PER_THREAD) ) {
                threaded(0, workloadSize, workload );
            }
            else sequential( workloadSize, workload );
        }

        /**
         *  This method will simply execute the provided {@link RangeWorkload} lambda sequentially
         *  with 0 as the start index and {@code workloadSize} as the exclusive range.       <br><br>
         *
         * @param workloadSize The workload size which will be passed to the provided {@link RangeWorkload} as second argument.
         * @param workload The {@link RangeWorkload} which will be executed sequentially.
         */
        public void sequential( int workloadSize, RangeWorkload workload ) { workload.execute( 0, workloadSize ); }


        /**
         *  Takes the provided range and divides it into multi-threaded workloads.
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
            _DIVIDER.parallelism( _PARALLELISM )
                    .threshold( PARALLELIZATION_THRESHOLD )
                    .divide( first, limit, rangeWorkload);
        }

    }

}