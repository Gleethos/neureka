package neureka.backend.standard.algorithms.internal;

@FunctionalInterface
public interface WithForward<F> {

    AndBackward<F> with( F forward );

}
