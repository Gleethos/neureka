package neureka.acceleration.host;

import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.AbstractDevice;
import neureka.acceleration.Device;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.Type;
import neureka.calculus.environment.subtypes.*;

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
    protected void _enqueue(Tsr[] tsrs, int d, OperationType type) {
        for (Tsr t : tsrs) t.setIsVirtual(false);
        if (type.supports(Activation.class) && !type.isIndexer()) _executor.activate(tsrs, d, type);
        else if (type.isOperation() && !type.isConvection()) _executor.broadcast(tsrs, d, type);
        else if (type.isConvection()) {
            if (type.identifier().contains(((char) 187) + "")) {
                _executor.convolve(new Tsr[]{tsrs[2], tsrs[1], tsrs[0]}, d, type);
            } else if (type.identifier().contains(((char) 171) + "")) {
                _executor.convolve(new Tsr[]{tsrs[0], tsrs[1], tsrs[2]}, d, type);
            } else {
                if (d >= 0) {
                    if (d == 0) tsrs[0] = tsrs[2];
                    else tsrs[0] = tsrs[1];
                } else {
                    _executor.convolve(tsrs, -1, type);
                }
            }
        } else if (type.isIndexer()) _executor.broadcast(tsrs, d, type);
    }

    @Override
    protected void _enqueue(Tsr t, double value, int d, OperationType type) {
        if (type.supports(Scalarization.class)) _executor.scalar(new Tsr[]{t, t}, value, d, type);
        else {
            int[] shape = new int[t.rank()];
            Arrays.fill(shape, 1);
            _enqueue(new Tsr[]{t, t, new Tsr(shape, value)}, d, type);
        }
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

    interface Range {
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
            _threaded(
                    tsrs[0].size(),
                    (start, end) ->
                            Kernel.activate(
                                    tsrs[0], start, end,
                                    type.get(Activation.class).getCreator().create(tsrs, d)
                            )
            );
        }

        public void broadcast(Tsr[] tsrs, int d, OperationType type)
        {
            _threaded(
                    tsrs[0].size(),
                    (start, end) ->
                            Kernel.broadcast(
                                    tsrs[0], tsrs[1], tsrs[2], d,
                                    start, end,
                                    type.get(Broadcast.class).getCreator().create(tsrs, d)
                            )
            );
        }

        public void convolve(Tsr[] tsrs, int d, OperationType type)
        {
            _threaded(
                    tsrs[0].size(),
                    (start, end) ->
                            Kernel.convolve(
                                    tsrs[0], tsrs[1], tsrs[2], d,
                                    start, end,
                                    type.get(Convolution.class).getCreator().create(tsrs, -1)
                            )
            );
        }

        public void scalar(Tsr[] tsrs, double scalar, int d, OperationType type)
        {
            _threaded(
                    tsrs[0].size(),
                    (start, end) ->
                            Kernel.activate(
                                    tsrs[0], start, end,
                                    type.get(Scalarization.class).getCreator().create(tsrs, scalar, d)
                            )
            );
        }

        //==============================================================================================================

        private void _threaded(int sze, Range range)
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