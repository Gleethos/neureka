package neureka.backend.standard.algorithms.internal;

import neureka.backend.standard.implementations.CLImplementation;

@FunctionalInterface
public interface AndBackward<F> {

    CLImplementation and( F backward );

}
