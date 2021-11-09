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

public class CPU extends AbstractDevice<Number>
{
    private static final Logger _LOG = LoggerFactory.getLogger( CPU.class );
    private static final CPU _INSTANCE;

    static {  _INSTANCE = new CPU();  }

    private final NativeExecutor _executor;
    private Set<Tsr<Number>> _tensors = Collections.newSetFromMap(new WeakHashMap<Tsr<Number>, Boolean>());

    private CPU() {
        super();
        _executor = new NativeExecutor();
    }

    public static CPU get() {
        return _INSTANCE;
    }

    public NativeExecutor getExecutor() {
        return _executor;
    }

    @Override
    protected boolean _approveExecutionOf( Tsr[] tensors, int d, Operation operation )
    {
        return true;
    }

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
    public Number valueFor( Tsr<Number> tensor, int index ) {
        return tensor.getValueAt( index );
    }

    @Override
    public Device restore( Tsr tensor ) {
        return this;
    }

    @Override
    public Device store( Tsr tensor ) {
        _tensors.add( tensor );
        return this;
    }

    @Override
    public Device store( Tsr tensor, Tsr parent ) {
        _tensors.add( tensor );
        _tensors.add( parent );
        return this;
    }

    @Override
    public boolean has( Tsr tensor ) {
        return _tensors.contains( tensor );
    }

    @Override
    public Device free( Tsr tensor ) {
        _tensors.remove( tensor );
        return this;
    }

    @Override
    public Device swap( Tsr former, Tsr replacement ) {
        return this;
    }

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

    public interface Range {
        void execute(int start, int end);
    }

    public class NativeExecutor
    {
        private final ThreadPoolExecutor _pool =
                (ThreadPoolExecutor) Executors.newFixedThreadPool(
                        Runtime.getRuntime().availableProcessors()
                );

        public ThreadPoolExecutor getPool() {
            return _pool;
        }

        //==============================================================================================================

        public void threaded( int sze, Range range )
        {
            int cores = _pool.getCorePoolSize() - _pool.getActiveCount();
            cores = ( cores == 0 ) ? 1 : cores;
            if ( sze >= 32 && ( ( sze / cores ) >= 8 ) ) {
                final int chunk = sze / cores;
                Future<?>[] futures = new Future[ cores ];
                for ( int i = 0; i < cores; i++ ) {
                    final int start = i * chunk;
                    final int end = ( i == cores - 1 ) ? sze : ( (i + 1) * chunk );
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
            } else range.execute(0, sze);
        }

    }

}