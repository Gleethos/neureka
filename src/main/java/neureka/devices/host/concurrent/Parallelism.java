/*<#LICENSE#>*/
package neureka.devices.host.concurrent;

import neureka.devices.host.machine.ConcreteMachine;

import java.util.function.IntSupplier;

/**
 * A set of standard levels of parallelism derived from the number of available cores and optionally capped by
 * reserving a specified amount of memory per thread. The info about available cores/threads/memory comes from
 * {@link ConcreteMachine#ENVIRONMENT}.
 */
public enum Parallelism implements IntSupplier {

    /**
     * The total number of threads (incl. hyperthreads)
     */
    THREADS(() -> ConcreteMachine.ENVIRONMENT.threads),
    /**
     * The number of CPU cores
     */
    CORES(() -> ConcreteMachine.ENVIRONMENT.cores),
    /**
     * The number of top level (L2 or L3) cache units
     */
    UNITS(() -> ConcreteMachine.ENVIRONMENT.units),
    /**
     * 8
     */
    EIGHT(() -> 8),
    /**
     * 4
     */
    FOUR(() -> 4),
    /**
     * 2
     */
    TWO(() -> 2),
    /**
     * 1
     */
    ONE(() -> 1);

    private final IntSupplier _supplier;

    Parallelism( final IntSupplier value ) {
        _supplier = value;
    }

    public int getAsInt() {
        return _supplier.getAsInt();
    }

}
