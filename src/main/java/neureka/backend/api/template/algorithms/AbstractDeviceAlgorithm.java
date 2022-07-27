package neureka.backend.api.template.algorithms;

import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Algorithm;
import neureka.backend.api.DeviceAlgorithm;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 *  This is a partial implementation of the {@link Algorithm} interface which implements
 *  the component system for implementation instances of the {@link ImplementationFor} interface.
 *  These components implement an algorithm for a specific {@link Algorithm}.
 *
 * @param <C> The type of the concrete extension of this class.
 */
public abstract class AbstractDeviceAlgorithm<C extends DeviceAlgorithm<C>>
extends AbstractAlgorithm
implements DeviceAlgorithm<C>
{
    private final Logger _LOG = LoggerFactory.getLogger(AbstractDeviceAlgorithm.class);

    protected final Map<Class<Device<?>>, ImplementationFor<Device<?>>> _implementations = new HashMap<>();

    public AbstractDeviceAlgorithm( String name ) { super( name ); }

    @Override
    public <D extends Device<?>, E extends ImplementationFor<D>> C setImplementationFor(
            Class<D> deviceClass, E implementation
    ) {
        if ( _implementations.containsKey( deviceClass ) )
            _LOG.info(
                    "Implementation for device '"+deviceClass.getSimpleName()+"' already defined!"
                );

        _implementations.put(
            (Class<Device<?>>) deviceClass,
            (ImplementationFor<Device<?>>) implementation
        );
        return (C) this;
    }

    @Override
    public <D extends Device<?>> ImplementationFor<D> getImplementationFor(Class<D> deviceClass )
    {
        ImplementationFor<D> found = (ImplementationFor<D>) _implementations.get( deviceClass );
        if ( found == null )
            for ( Class<Device<?>> type : _implementations.keySet() )
                if ( type.isAssignableFrom(deviceClass) )
                    return (ImplementationFor<D>) _implementations.get(type);

        return found;
    }

    @Override
    public String toString() {
        String algorithmString = getClass().getSimpleName()+"@"+Integer.toHexString(hashCode());
        String implementations = _implementations.keySet().stream().map(Class::getSimpleName).collect(Collectors.joining(","));
        algorithmString = ( algorithmString + "[name=" + getName() + ",support=[" + implementations + "]]" );
        return algorithmString;
    }

}
