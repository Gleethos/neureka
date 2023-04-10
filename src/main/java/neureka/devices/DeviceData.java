package neureka.devices;

import neureka.Data;
import neureka.ndim.config.NDConfiguration;

/**
 *  A sub-interface of the {@link Data} interface providing
 *  more device specific methods.
 *
 * @param <V> The data type of the data.
 */
public interface DeviceData<V> extends Data<V>
{
    DeviceData<V> withNDConf( NDConfiguration ndc );
}
