/*<#LICENSE#>*/
package neureka.devices.host.concurrent;

import neureka.devices.host.CPU;
import neureka.devices.host.machine.ConcreteMachine;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.function.IntSupplier;

/**
 *  An API for registering workloads which will be divided into smaller workloads so that they can
 *  be executed efficiently by a thread pool... <br>
 *  This is a library internal class, do not depend on this.
 */
public abstract class WorkScheduler {

    public WorkScheduler() { super(); }

    /**
     * Synchronous execution - wait until it's finished.
     *
     * @param first The first index, in a range, to include.
     * @param limit The first index NOT to include - last (excl.) index in a range.
     * @param threshold The work size threshold.
     */
    public final void invoke(
            final ExecutorService executor,
            final int first,
            final int limit,
            final int threshold
    ) {
        int availableWorkers = ConcreteMachine.ENVIRONMENT.threads;
        Divider.devide(
            executor,
            first,
            limit,
            threshold,
            availableWorkers,
            this::_work
        );
    }

    protected abstract void _work(final int first, final int limit);


    /**
     *  Divides workloads until they can be processed efficiently
     *  and then submits them to a thread pool for execution... <br>
     *  This is a library internal class, do not depend on this.
     */
    public static final class Divider
    {
        private final ExecutorService _executor;

        private IntSupplier _parallelism = Parallelism.THREADS;

        private int _threshold = 128;

        public Divider( final ExecutorService executor ) {
            super();
            _executor = executor;
        }

        public void divide( final int limit, final CPU.RangeWorkload rangeWorkload) {
            this.divide(0, limit, rangeWorkload);
        }

        public void divide( final int first, final int limit, final CPU.RangeWorkload rangeWorkload) {
            devide(
                    _executor,
                    first,
                    limit,
                    _threshold,
                    _parallelism.getAsInt(),
                    rangeWorkload
            );
        }

        public Divider parallelism(
                final IntSupplier parallelism
        ) {
            if ( parallelism != null ) {
                _parallelism = parallelism;
            }
            return this;
        }

        public Divider threshold( final int threshold ) {
            _threshold = threshold;
            return this;
        }

        private static void devide(
                final ExecutorService executor,
                final int start,
                final int end,
                final int threshold,
                final int workers,
                final CPU.RangeWorkload rangeWorkload
        ) {
            int workload = end - start;

            if ( workload > threshold && workers > 1 ) {

                int split = start + workload / 2;
                int nextWorkers = workers / 2;

                Future<?> firstPart = executor.submit( () -> devide(executor, start, split, threshold, nextWorkers, rangeWorkload) );
                Future<?> secondPart = executor.submit( () -> devide(executor, split, end, threshold, nextWorkers, rangeWorkload) );

                try {
                    firstPart.get();
                    secondPart.get();
                } catch ( final InterruptedException | ExecutionException cause ) {
                    throw new RuntimeException(cause);
                }
            }
            else
                rangeWorkload.execute(start, end);
        }

    }

}
