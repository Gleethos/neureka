package neureka.devices.host;

import neureka.Neureka;
import neureka.Tsr;
import neureka.devices.AbstractDevice;
import neureka.devices.Device;
import neureka.devices.host.execution.HostExecutor;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.ExecutionCall;

import java.util.Collection;
import java.util.concurrent.*;

public class HostCPU extends AbstractDevice<Number>
{
    private static final HostCPU _instance;

    static {  _instance = new HostCPU();  }

    private final NativeExecutor _executor;

    private HostCPU() {
        _executor = new NativeExecutor();
    }

    public static HostCPU instance() {
        return _instance;
    }

    public NativeExecutor getExecutor() {
        return _executor;
    }

    @Override
    protected void _execute(Tsr[] tensors, int d, OperationType type)
    {
        ExecutionCall<HostCPU> call =
                new ExecutionCall<>(
                        this,
                        tensors,
                        d,
                        type
                );
        call.getImplementation().getExecutor(HostExecutor.class).getExecution().run(call);
    }

    @Override
    public void dispose() {
        _executor.getPool().shutdown();
    }

    @Override
    public Device restore(Tsr tensor) {
        return this;
    }

    @Override
    public Device store(Tsr tensor) {
        return this;
    }

    @Override
    public Device store(Tsr tensor, Tsr parent) {
        return this;
    }

    @Override
    public boolean has( Tsr tensor ) {
        return false;
    }

    @Override
    public Device free(Tsr tensor) {
        return this;
    }

    @Override
    public Device overwrite64(Tsr tensor, double[] value) {
        return this;
    }

    @Override
    public Device overwrite32(Tsr tensor, float[] value) {
        return this;
    }

    @Override
    public Device swap(Tsr former, Tsr replacement) {
        return this;
    }

    @Override
    public double[] value64f( Tsr tensor ) {
        return tensor.value64();
    }

    @Override
    public float[] value32f( Tsr tensor ) {
        return tensor.value32();
    }

    @Override
    public double value64f(Tsr tensor, int index) {
        return tensor.value64(index);
    }

    @Override
    public float value32f(Tsr tensor, int index) {
        return tensor.value32(index);
    }

    @Override
    public Collection<Tsr<Number>> getTensors() {
        return null;
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
                Future[] futures = new Future[cores];
                for (int i = 0; i < cores; i++) {
                    final int start = i * chunk;
                    final int end = ( i == cores - 1 ) ? sze : ( (i + 1) * chunk );
                    Neureka neureka = Neureka.instance();
                    futures[ i ] = _pool.submit(() -> {
                        Neureka.setContext( neureka );
                        range.execute(start, end);
                    });
                }
                for (Future f : futures) {
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