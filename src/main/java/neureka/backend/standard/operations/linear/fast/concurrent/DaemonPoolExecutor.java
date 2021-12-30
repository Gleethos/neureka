/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.concurrent;

import neureka.backend.standard.operations.linear.fast.machine.ConcreteMachine;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public final class DaemonPoolExecutor extends ThreadPoolExecutor {

    private static final AtomicInteger COUNTER = new AtomicInteger();
    private static final ThreadGroup GROUP = new ThreadGroup("neureka-daemon-group");

    static final DaemonPoolExecutor INSTANCE = new DaemonPoolExecutor(
                                                    ConcreteMachine.ENVIRONMENT.units,
                                                    Integer.MAX_VALUE,
                                        5L,
                                                    TimeUnit.SECONDS,
                                                    new SynchronousQueue<Runnable>(),
                                                    DaemonPoolExecutor.newThreadFactory("neureka-daemon-")
                                                    );

    /**
     * @see AbstractExecutorService#submit(Callable)
     */
    public static <T> Future<T> invoke(final Callable<T> task) {
        return INSTANCE.submit(task);
    }

    /**
     * @see AbstractExecutorService#submit(Runnable)
     */
    public static Future<?> invoke(final Runnable task) {
        return INSTANCE.submit(task);
    }

    /**
     * @see AbstractExecutorService#submit(Runnable, Object)
     */
    public static <T> Future<T> invoke(final Runnable task, final T result) {
        return INSTANCE.submit(task, result);
    }

    public static ThreadFactory newThreadFactory(final String name) {
        return DaemonPoolExecutor.newThreadFactory(GROUP, name);
    }

    public static ThreadFactory newThreadFactory(final ThreadGroup group, final String name) {

        String prefix = name.endsWith("-") ? name : name + "-";

        return target -> {
            Thread thread = new Thread(group, target, prefix + DaemonPoolExecutor.COUNTER.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        };
    }

    DaemonPoolExecutor(
            final int corePoolSize,
            final int maximumPoolSize,
            final long keepAliveTime,
            final TimeUnit unit,
            final BlockingQueue<Runnable> workQueue,
            final ThreadFactory threadFactory
    ) {
        super(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue, threadFactory);
    }

}
