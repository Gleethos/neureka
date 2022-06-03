package neureka.backend.main.algorithms.internal;

import neureka.backend.main.implementations.CLImplementation;

@FunctionalInterface
public interface AndBackward<F> {

    CLImplementation and( F backward );

}
