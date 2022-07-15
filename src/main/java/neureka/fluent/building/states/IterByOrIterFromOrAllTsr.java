package neureka.fluent.building.states;

import neureka.Tsr;
import neureka.ndim.Filler;

import java.util.List;

public interface IterByOrIterFromOrAllTsr<V> extends IterByOrIterFromOrAll<V>
{
    /** {@inheritDoc} */
    @Override Tsr<V> andFill( V... values );

    /** {@inheritDoc} */
    @Override default Tsr<V> andFill( List<V> values ) {
        return this.andFill((V[])values.toArray());
    }

    /** {@inheritDoc} */
    @Override Tsr<V> andWhere( Filler<V> filler );

    /** {@inheritDoc} */
    @Override ToForTsr<V> andFillFrom( V index );

    /** {@inheritDoc} */
    @Override Tsr<V> all( V value );

    /** {@inheritDoc} */
    @Override Tsr<V> andSeed( Object seed );

}
