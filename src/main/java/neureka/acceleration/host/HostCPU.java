package neureka.acceleration.host;

import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.AbstractDevice;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;
import neureka.calculus.environment.executors.*;

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
        for ( Tsr t : tsrs ) t.setIsVirtual(false);

        OperationTypeImplementation.ExecutionCall<HostCPU> call =
                new OperationTypeImplementation.ExecutionCall<>(
                        this,
                        tsrs,
                        d,
                        type
                );
        call.getExecutor().getExecution(HostExecutor.class).getExecution().call(call);
    }

    @Override
    protected void _enqueue(Tsr t, double value, int d, OperationType type) {
        //if (type.supportsImplementation(Scalarization.class)) {
        //    OperationTypeImplementation.ExecutionCall<HostCPU> call =
        //            new OperationTypeImplementation.ExecutionCall<>(
        //                    this,
        //                    new Tsr[]{t, t},
        //                    d,
        //                    type
        //            );
        //    call.getExecutor().callImplementationFor(call);
        //    _executor.scalar(new Tsr[]{t, t}, value, d, type);
        //}
        //else {
            int[] shape = new int[t.rank()];
            Arrays.fill(shape, 1);
            _enqueue(new Tsr[]{t, t, new Tsr(shape, value)}, d, type);
        //}
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
    public double[] value64Of(Tsr tensor) {
        return tensor.value64();
    }

    @Override
    public float[] value32Of(Tsr tensor) {
        return tensor.value32();
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
                (ThreadPoolExecutor) Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        public ThreadPoolExecutor getPool() {
            return _pool;
        }

        public void activate(Tsr[] tsrs, int d, OperationType type)
        {
            threaded(
                    tsrs[0].size(),
                    (start, end) ->
                            Activation.activate(
                                    tsrs[0], start, end,
                                    type.getImplementation(Activation.class).getCreator().create(tsrs, d)
                            )
            );
        }

        public void broadcast(Tsr[] tsrs, int d, OperationType type)
        {
            threaded(
                    tsrs[0].size(),
                    (start, end) ->
                            Broadcast.broadcast(
                                    tsrs[0], tsrs[1], tsrs[2], d,
                                    start, end,
                                    type.getImplementation(Broadcast.class).getCreator().create(tsrs, d)
                            )
            );
        }

        public void convolve(Tsr[] tsrs, int d, OperationType type)
        {
            threaded(
                    tsrs[0].size(),
                    (start, end) ->
                            Convolution.convolve(
                                    tsrs[0], tsrs[1], tsrs[2], d,
                                    start, end,
                                    type.getImplementation(Convolution.class).getCreator().create(tsrs, -1)
                            )
            );
        }

        public void scalar(Tsr[] tsrs, double scalar, int d, OperationType type)
        {
            threaded(
                    tsrs[0].size(),
                    (start, end) ->
                            Scalarization.scalarize(
                                    tsrs[0], start, end,
                                    type.getImplementation(Scalarization.class).getCreator().create(tsrs, scalar, d)
                            )
            );
        }

        //==============================================================================================================

        public void threaded(int sze, Range range)
        {
            int cores = _pool.getCorePoolSize() - _pool.getActiveCount();
            cores = (cores == 0) ? 1 : cores;
            if (sze >= 32 && ((sze / cores) >= 8)) {
                final int chunk = sze / cores;
                Future[] futures = new Future[cores];
                for (int i = 0; i < cores; i++) {
                    final int start = i * chunk;
                    final int end = (i == cores - 1) ? sze : ((i + 1) * chunk);
                    Neureka neureka = Neureka.instance();
                    futures[i] = _pool.submit(() -> {
                        Neureka.setContext(Thread.currentThread(), neureka);
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