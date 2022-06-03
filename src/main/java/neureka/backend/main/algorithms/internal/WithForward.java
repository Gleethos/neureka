package neureka.backend.main.algorithms.internal;

@FunctionalInterface
public interface WithForward<F> {

    AndBackward<F> with( F forward );

}
