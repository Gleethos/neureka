package neureka.backend.api.algorithms;

import neureka.backend.api.Algorithm;
import neureka.backend.api.ImplementationFor;
import neureka.devices.Device;

import java.util.HashMap;
import java.util.Map;

public abstract class AbstractBaseAlgorithm<FinalType extends Algorithm<FinalType>> implements Algorithm<FinalType>
{
    /**
     *  This is the name of this {@link Algorithm}
     *  which may be used as variable names in OpenCL kernels or other backends.
     *  Therefore this name is expected to be void of any spaces
     *  or non numeric and alphabetic characters.
     */
    private final String _name;

    protected final Map< Class< Device<?> >, ImplementationFor< Device<?> >> _implementations = new HashMap<>();

    public AbstractBaseAlgorithm(String name) { _name = name; }

    //---

    @Override
    public <D extends Device<?>, E extends ImplementationFor<D>> FinalType setImplementationFor( Class<D> deviceClass, E implementation ) {
        _implementations.put(
                (Class<Device<?>>) deviceClass,
                (ImplementationFor<Device<?>>) implementation
        );
        return (FinalType) this;
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

    /**
     *  This method returns the name of this {@link Algorithm}
     *  which may be used as variable names in OpenCL kernels or other backends.
     *  Therefore this name is expected to be void of any spaces
     *  or non numeric and alphabetic characters.
     *
     * @return The name of this {@link Algorithm}.
     */
    public String getName() { return this._name; }
}
