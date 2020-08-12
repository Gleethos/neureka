package neureka.acceleration.host;

import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.AbstractDevice;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;

import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.*;

public class HostCPU extends AbstractDevice
{
    private static final HostCPU _instance;

    static {
        _instance = new HostCPU();
    }

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
    protected void _enqueue(Tsr[] tsrs, int d, OperationType type)
    {
        ExecutionCall<HostCPU> call =
                new ExecutionCall<>(
                        this,
                        tsrs,
                        d,
                        type
                );
        call.getImplementation().getExecutor(HostExecutor.class).getExecution().call(call);
    }

    @Override
    protected void _enqueue(Tsr t, double value, int d, OperationType type) {
        int[] shape = new int[t.rank()];
        Arrays.fill(shape, 1);
        _enqueue(new Tsr[]{t, t, new Tsr(shape, value)}, d, type);
    }

    @Override
    public void dispose() {
        _executor.getPool().shutdown();
    }

    @Override
    public Device get(Tsr tensor) {
        return this;
    }

    @Override
    public Device add(Tsr tensor) {
        return this;
    }

    @Override
    public Device add(Tsr tensor, Tsr parent) {
        return this;
    }

    @Override
    public boolean has(Tsr tensor) {
        return false;
    }

    @Override
    public Device rmv(Tsr tensor) {
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
        return 0;
    }

    @Override
    public float value32f(Tsr tensor, int index) {
        return 0;
    }

    @Override
    public Collection<Tsr> tensors() {
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
                    futures[i] = _pool.submit(() -> {
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