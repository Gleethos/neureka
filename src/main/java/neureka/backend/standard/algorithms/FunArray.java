package neureka.backend.standard.algorithms;

import neureka.backend.api.Fun;

public interface FunArray<T extends Fun> {

    T get( int derivativeIndex );

    Class<T> getType();

}
