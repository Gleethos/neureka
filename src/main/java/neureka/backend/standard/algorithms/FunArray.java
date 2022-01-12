package neureka.backend.standard.algorithms;

public interface FunArray<T extends Fun> {

    T get( int derivativeIndex );

    Class<T> getType();

}
