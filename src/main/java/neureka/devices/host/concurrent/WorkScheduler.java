/*<#LICENSE#>*/
package neureka.devices.host.concurrent;

import neureka.devices.host.machine.ConcreteMachine;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.function.IntSupplier;

/**
 */
public abstract class WorkScheduler {

    @FunctionalInterface
    public interface Worker {

        void work(final int start, final int end);

    }

    public static final class Divider {

        private final ExecutorService _executor;

        private IntSupplier _parallelism = Parallelism.THREADS;

        private int _threshold = 128;

        Divider(final ExecutorService executor) {
            super();
            _executor = executor;
        }

        public void divide(final int limit, final Worker worker) {
            this.divide(0, limit, worker);
        }

        public void divide(final int first, final int limit, final Worker worker) {
            WorkScheduler._call(_executor, first, limit, _threshold, _parallelism.getAsInt(), worker);
        }

        public Divider parallelism(
                final IntSupplier parallelism
        ) {
            if ( parallelism != null ) {
                _parallelism = parallelism;
            }
            return this;
        }

        public Divider threshold(final int threshold) {
            _threshold = threshold;
            return this;
        }

    }

    private static void _call(
            final ExecutorService executor,
            final int start,
            final int end,
            final int threshold,
            final int workers,
            final Worker worker
    ) {
        int workload = end - start;

        if ( workload > threshold && workers > 1 ) {

            int split = start + workload / 2;
            int nextWorkers = workers / 2;

            Future<?> firstPart = executor.submit(() -> WorkScheduler._call(executor, start, split, threshold, nextWorkers, worker));
            Future<?> secondPart = executor.submit(() -> WorkScheduler._call(executor, split, end, threshold, nextWorkers, worker));

            try {
                firstPart.get();
                secondPart.get();
            } catch (final InterruptedException | ExecutionException cause) {
                throw new RuntimeException(cause);
            }
        }
        else
            worker.work(start, end);
    }

    public WorkScheduler() {
        super();
    }

    /**
     * Synchronous execution - wait until it's finished.
     *
     * @param first The first index, in a range, to include.
     * @param limit The first index NOT to include - last (excl.) index in a range.
     */
    public final void invoke(
            final int first,
            final int limit,
            final int threshold
    ) {
        int availableWorkers = ConcreteMachine.ENVIRONMENT.threads;
        _call(
                DaemonPoolExecutor.INSTANCE,
                first,
                limit,
                threshold,
                availableWorkers,
                this::_work
        );
    }

    protected abstract void _work(final int first, final int limit);

}