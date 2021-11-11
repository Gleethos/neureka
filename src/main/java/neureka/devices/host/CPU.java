package neureka.devices.host;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.devices.AbstractDevice;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/**
 *  This is a singleton class which simply represents the CPU as a {@link Device}.
 *  Tensors stored on the {@link CPU} simply reside in the JVM heap.
 */
public class CPU extends AbstractDevice<Number>
{
    private static final Logger _LOG = LoggerFactory.getLogger( CPU.class );
    private static final CPU _INSTANCE;

    static {  _INSTANCE = new CPU();  }

    private final JVMExecutor _executor;
    private final Set<Tsr<Number>> _tensors = Collections.newSetFromMap(new WeakHashMap<Tsr<Number>, Boolean>());

    private CPU() {
        super();
        _executor = new JVMExecutor();
    }

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
     *  however it differs in that the {@link JVMExecutor} is processing {@link Range} lambdas
     *  instead of simply exposing a single index or concrete elements for a given workload size.
     *
     * @return A parallel range based execution API running on the JVM.
     */
    public JVMExecutor getExecutor() {
        return _executor;
    }

    @Override
    protected boolean _approveExecutionOf( Tsr[] tensors, int d, Operation operation ) { return true; }

    @Override
    public void dispose() {
        _executor.getPool().shutdown();
    }

    @Override
    public Device<Number> write(Tsr<Number> tensor, Object value) {
        return this;
    }

    @Override
    public Object valueFor( Tsr<Number> tensor ) {
        return tensor.getValue();
    }

    @Override
    public Number valueFor( Tsr<Number> tensor, int index ) { return tensor.getValueAt( index ); }

    @Override
    public CPU restore( Tsr<Number> tensor ) { return this; }

    @Override
    public CPU store( Tsr tensor ) {
        _tensors.add( tensor );
        return this;
    }

    @Override
    public CPU store( Tsr tensor, Tsr parent ) {
        _tensors.add( tensor );
        _tensors.add( parent );
        return this;
    }

    @Override
    public boolean has( Tsr tensor ) {
        return _tensors.contains( tensor );
    }

    @Override
    public CPU free( Tsr tensor ) {
        _tensors.remove( tensor );
        return this;
    }

    @Override
    public CPU swap( Tsr former, Tsr replacement ) { return this; }

    @Override
    public Collection<Tsr<Number>> getTensors() {
        return _tensors;
    }

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
    public boolean update( OwnerChangeRequest<Tsr<Number>> changeRequest ) {
        super.update( changeRequest );
        return false; // This type of device can not be a component simply because it is the default device
        //super.update( changeRequest );
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
    public int getCoreCount() {
        return Runtime.getRuntime().availableProcessors();
    }

    public interface Range {
        void execute(int start, int end);
    }

    /**
     *  The {@link JVMExecutor} offers a similar functionality as the parallel stream API,
     *  however it differs in that the {@link JVMExecutor} is processing {@link Range} lambdas
     *  instead of simply exposing a single index or concrete elements for a given workload size.
     *  This means that a {@link Range} lambda will be called with the work range of a single worker thread
     *  processing its current workload.
     *  This range is dependent on the number of available threads as well as the size of the workload.
     *  If the workload is very small, then the current main thread will process the entire workload range
     *  whereas the underlying {@link ThreadPoolExecutor} will not be used to avoid unnecessary overhead.
     */
    public static class JVMExecutor
    {
        private final ThreadPoolExecutor _pool =
                (ThreadPoolExecutor) Executors.newFixedThreadPool(
                        Runtime.getRuntime().availableProcessors()
                );

        public ThreadPoolExecutor getPool() { return _pool; }

        //==============================================================================================================

        public void threaded( int size, Range range )
        {
            int cores = _pool.getCorePoolSize() - _pool.getActiveCount();
            cores = ( cores == 0 ) ? 1 : cores;
            if ( size >= 32 && ( ( size / cores ) >= 8 ) ) {
                final int chunk = size / cores;
                Future<?>[] futures = new Future[ cores ];
                for ( int i = 0; i < cores; i++ ) {
                    final int start = i * chunk;
                    final int end = ( i == cores - 1 ) ? size : ( (i + 1) * chunk );
                    Neureka neureka = Neureka.get();
                    futures[ i ] = _pool.submit(() -> {
                        Neureka.set( neureka ); // This ensures that the threads in the pool have the same settings!
                        range.execute( start, end );
                    });
                }
                for ( Future<?> f : futures ) {
                    try {
                        f.get();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } catch (ExecutionException e) {
                        e.printStackTrace();
                    }
                }
            }
            else range.execute(0, size);
        }

    }

}