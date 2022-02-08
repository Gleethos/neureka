package neureka.backend.standard.algorithms.internal;

public interface WithForward<F> {

    AndBackward<F> with( F forward );

}
