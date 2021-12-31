/*<#LICENSE#>*/
package neureka.devices.host.machine;

/**
 * Stuff common to {@link Hardware} and {@link ConcreteMachine}.
 *
 */
public abstract class CommonMachine extends BasicMachine {

    static final long K = 1024L;

    public final String architecture;//x86_64

    /**
     * The size of one top level (L3 or L2) cache unit in bytes.
     */
    public final long cache;
    /**
     * The total number of processor cores.
     */
    public final int cores;
    /**
     * The number of top level (L3 or L2) cache units. With L3 cache defined this corresponds to the number of
     * CPU:s.
     */
    public final int units;

    protected CommonMachine(final Hardware hardware, final Runtime runtime) {

        super(Math.min(hardware.memory, runtime.maxMemory()), Math.min(hardware.threads, runtime.availableProcessors()));

        architecture = hardware.architecture;

        cache = hardware.cache;

        cores = hardware.cores;
        units = hardware.units;
    }

    /**
     * <code>new MemoryThreads[] { SYSTEM, L3, L2, L1 }</code> or
     * <code>new MemoryThreads[] { SYSTEM, L2, L1 }</code> or in worst case
     * <code>new MemoryThreads[] { SYSTEM, L1 }</code>
     */
    protected CommonMachine(final String architecture, final BasicMachine[] levels) {

        super(levels[0].memory, levels[0].threads);

        this.architecture = architecture;

        cores = threads / levels[levels.length - 1].threads;
        cache = levels[1].memory;
        units = threads / levels[1].threads;
    }

    @Override
    public boolean equals(final Object obj) {
        if (this == obj) {
            return true;
        }
        if (!super.equals(obj)) {
            return false;
        }
        if (!(obj instanceof CommonMachine)) {
            return false;
        }
        CommonMachine other = (CommonMachine) obj;
        if (architecture == null) {
            if (other.architecture != null) {
                return false;
            }
        } else if (!architecture.equals(other.architecture)) {
            return false;
        }
        if (cache != other.cache) {
            return false;
        }
        if (cores != other.cores) {
            return false;
        }
        if (units != other.units) {
            return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = (prime * result) + ((architecture == null) ? 0 : architecture.hashCode());
        result = (prime * result) + (int) (cache ^ (cache >>> 32));
        result = (prime * result) + cores;
        result = (prime * result) + units;
        return result;
    }

}
