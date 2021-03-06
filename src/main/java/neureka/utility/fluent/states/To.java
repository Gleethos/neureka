package neureka.utility.fluent.states;

import neureka.Tsr;

/**
 *  This step in the call transition graph of the fluent builder API is a followup call
 *  from the {@link IterByOrIterFromOrAll#iterativelyFilledFrom(Object)} method which
 *  expects a range to be specified whose values will be used to populate the {@link Tsr} instance.
 *  This method allows one to set the end point of the range whose start has previously between set
 *  in the {@link IterByOrIterFromOrAll#iterativelyFilledFrom(Object)} method...
 *
 * @param <V>
 */
public interface To<V> {

    /**
     *  This step in the call transition graph of the fluent builder API is a followup call
     *  from the {@link IterByOrIterFromOrAll#iterativelyFilledFrom(Object)} method which
     *  expects a range to be specified whose values will be used to populate the {@link Tsr} instance.
     *  This method allows one to set the end point of the range whose start has previously between set
     *  in the {@link IterByOrIterFromOrAll#iterativelyFilledFrom(Object)} method...
     *
     * @param index The end point of the range previously specified in {@link IterByOrIterFromOrAll#iterativelyFilledFrom(Object)}.
     * @return The last step in the call transition graph of the fluent builder API for building range based {@link Tsr} instances.
     */
    Step<V> to( V index );

}
