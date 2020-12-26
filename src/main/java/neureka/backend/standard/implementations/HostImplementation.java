package neureka.backend.standard.implementations;

import neureka.backend.api.implementations.AbstractImplementationFor;
import neureka.backend.api.implementations.ImplementationFor;
import neureka.devices.host.HostCPU;

/**
 * This class is a wrapper class for the ImplementationFor&lt;HostCPU&gt; interface
 * which enables a functional style of implementing the backend API!<br>
 * It is used merely as a simple formality and implementation type specification.
 *
 * used to properly call a HostCPU instance via the
 * ImplementationFor&lt;HostCPU&gt; lambda implementation
 * receiving an instance of the ExecutionCall class.
 */
public class HostImplementation extends AbstractImplementationFor<HostCPU>
{
    public HostImplementation( ImplementationFor<HostCPU> creator, int arity )
    {
        super( creator, arity );
    }
}
