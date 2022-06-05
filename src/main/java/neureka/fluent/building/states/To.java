package neureka.fluent.building.states;

import neureka.Tsr;

/**
 *  This step in the call transition graph of the fluent builder API is a followup call
 *  from the {@link IterByOrIterFromOrAll#andFillFrom(Object)} method which
 *  expects a range to be specified whose values will be used to populate the {@link Tsr} instance.
 *  This method allows one to set the end point of the range whose start has previously between set
 *  in the {@link IterByOrIterFromOrAll#andFillFrom(Object)} method...
 *
 * @param <V> The type parameter of the second and last range index.
 */
public interface To<V> {

    /**
     *  This step in the call transition graph of the fluent builder API is a followup call
     *  from the {@link IterByOrIterFromOrAll#andFillFrom(Object)} method which
     *  expects a range to be specified whose values will be used to populate the {@link Tsr} instance.
     *  This method allows one to set the end point of the range whose start has previously between set
     *  in the {@link IterByOrIterFromOrAll#andFillFrom(Object)} method...
     *
     * @param index The end point of the range previously specified in {@link IterByOrIterFromOrAll#andFillFrom(Object)}.
     * @return The last step in the call transition graph of the fluent builder API for building range based {@link Tsr} instances.
     */
    Step<V> to( V index );

}
