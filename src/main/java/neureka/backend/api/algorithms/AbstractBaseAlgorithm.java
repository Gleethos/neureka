package neureka.backend.api.algorithms;

import neureka.backend.api.Algorithm;
import neureka.backend.api.ImplementationFor;
import neureka.devices.Device;

import java.util.HashMap;
import java.util.Map;

/**
 *  This is a partial implementation of the {@link Algorithm} interface which implements
 *  the component system for implementation instances of the {@link ImplementationFor} interface.
 *  These components implement an algorithm for a specific {@link Algorithm}.
 *
 * @param <C> The type of the concrete extension of this class.
 */
public abstract class AbstractBaseAlgorithm<C extends Algorithm<C>> implements Algorithm<C>
{
    /**
     *  This is the name of this {@link Algorithm}
     *  which may be used as variable names in OpenCL kernels or other backends.
     *  Therefore this name is expected to be void of any spaces
     *  or non numeric and alphabetic characters.
     */
    private final String _name;

    protected final Map<Class<Device<?>>, ImplementationFor<Device<?>>> _implementations = new HashMap<>();

    public AbstractBaseAlgorithm( String name ) { _name = name; }


    /**
     *  This method returns the name of this {@link Algorithm}
     *  which may be used as variable names in OpenCL kernels or other backends.
     *  Therefore this name is expected to be void of any spaces
     *  or non numeric and alphabetic characters.
     *
     * @return The name of this {@link Algorithm}.
     */
    public String getName() { return this._name; }

    @Override
    public <D extends Device<?>, E extends ImplementationFor<D>> C setImplementationFor(Class<D> deviceClass, E implementation ) {
        _implementations.put(
                (Class<Device<?>>) deviceClass,
                (ImplementationFor<Device<?>>) implementation
        );
        return (C) this;
    }

    @Override
    public <D extends Device<?>> ImplementationFor<D> getImplementationFor( Class<D> deviceClass ) {
        ImplementationFor<D> found = (ImplementationFor<D>) _implementations.get( deviceClass );
        if ( found == null ) {
            for ( Class<Device<?>> type : this._implementations.keySet() ) {
                if ( type.isAssignableFrom(deviceClass) ) return (ImplementationFor<D>) _implementations.get(type);
            }
        }
        return found;
    }

}
