package neureka.backend.main.implementations;

import neureka.backend.api.ImplementationFor;
import neureka.backend.api.template.implementations.AbstractImplementationFor;
import neureka.devices.host.CPU;

/**
 * This class is a wrapper class for the {@link ImplementationFor}&lt;{@link CPU}&gt; interface
 * which enables a functional style of implementing the backend API!<br>
 * It is used merely as a simple formality and implementation type specification.
 *
 * used to properly call a {@link CPU} instance via the
 * {@link ImplementationFor}&lt;{@link CPU}&gt; lambda implementation
 * receiving an instance of the ExecutionCall class.
 */
public class CPUImplementation extends AbstractImplementationFor<CPU>
{
    public static AndImplementation withArity( int arity ) { return lambda -> new CPUImplementation( lambda, arity ); }

    private CPUImplementation( ImplementationFor<CPU> creator, int arity ) { super( creator, arity ); }

    /**
     *  This is represents the second step in the simple builder API for {@link CPUImplementation} instances.
     *  The starting point for constructing a new instance is {@link #withArity(int)}, a static factory method.
     */
    public interface AndImplementation {
        CPUImplementation andImplementation( ImplementationFor<CPU> creator );
    }
}
