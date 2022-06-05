package neureka.fluent.building.states;

import neureka.Tsr;

/**
 *  This interface defines the last step in the call transition graph of the fluent builder API when
 *  building a {@link Tsr} instance populated based on the values within a defined range.
 *  This method embodies the last part of this range definition which consists of the chained
 *  methods {@link IterByOrIterFromOrAll#andFillFrom(Object)}, {@link To#to(Object)}
 *  and lastly the method defined in this interface, namely: {@link #step(double)}.
 *
 * @param <V> The type of the values wrapped by the {@link Tsr} which is about to be instantiated.
 */
public interface Step<V>
{
    /**
     *  This is the last step in the call transition graph of the fluent builder API when
     *  building a {@link Tsr} instance populated based on the values within a defined range.
     *  This method embodies the last part of this range definition which consists of the chained
     *  methods {@link IterByOrIterFromOrAll#andFillFrom(Object)}, {@link To#to(Object)}
     *  and lastly this very method {@link #step(double)}.
     *  This method allows one to set the step size used to space the entries within
     *  the previously defined span between what has
     *  been passed to {@link IterByOrIterFromOrAll#andFillFrom(Object)}
     *  and what has been passed to {@link To#to(Object)}...
     *
     * @param size The size of the step within the range defined by this fluent builder API.
     * @return A new {@link Tsr} instance whose contents are filled based on the provided range.
     */
    Tsr<V> step( double size );

}
