package neureka.backend.standard.algorithms;

public interface WithForward<F> {

    AndBackward<F> with( F forward );

}
