package neureka.ndim.config.types;

import neureka.ndim.config.AbstractNDC;

/**
 *  An abstract class for {@link neureka.ndim.config.NDConfiguration}s which are representing
 *  tensors of rank 1, meaning the name of this class translates to "Dimension-1-Configuration".
 */
public abstract class D1C extends AbstractNDC
{
    public abstract int indexOfIndices( int d1 );
}
