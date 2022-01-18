package neureka.backend.standard.memory;

import neureka.Tsr;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 *  This class validates the states of tensors with respect to memory management
 *  before and after a lambda executes a function or some kind of algorithm on said tensors.
 *  This validity refers to the {@link Tsr#isIntermediate()} flag, whose state should
 *  adhere to strict rules in order to allow for safe deletion of tensors.
 *  The lambda wrapped by this may be a {@link neureka.calculus.Function} call or a lower level
 *  procedure defined a {@link neureka.backend.api.Algorithm} implementation.
 *  <br><br>
 *  <b>Warning! This is an internal class. Do not depend on it.</b>
 */
public class MemValidator {

    private final Tsr<?> _result;
    private final boolean _wronglyIntermediate;
    private final boolean _wronglyNonIntermediate;

    /**
     * @param inputs The inputs used by the {@link Supplier} implementation to provide a result.
     * @param resultProvider The callback providing the result which ought to be validated.
     * @return The {@link MemValidator} which ought to validate the provided result.
     */
    public static MemValidator forInputs(Tsr<?>[] inputs, Supplier<Tsr<?>> resultProvider ) {
        return new MemValidator( inputs, resultProvider );
    }

    private MemValidator( Tsr<?>[] tensors, Supplier<Tsr<?>> execution ) {
        /*
            Now before calling the function we will do a snapshot of the inputs
            in order to later on verify the output validity with respect
            to the 'intermediate' flag.
        */
        Tsr<?>[] inputs = tensors.clone();
        Boolean[] areIntermediates  = Arrays.stream(tensors).map(Tsr::isIntermediate).toArray(Boolean[]::new);
        Boolean[] gradIntermediates = Arrays.stream(tensors).map( t -> (t.hasGradient() && t.getGradient().isIntermediate()) ).toArray(Boolean[]::new);
        /*
            Finally, we dispatch the call to the function implementation to get as result!
        */
        Tsr<?> result = execution.get();
        /*
            After that we analyse the validity of the result
            with respect to memory safety!
            We expect internally created tensors to be flagged as 'intermediate':
        */
        if ( result != null ) {
            /*
                First we check if the result tensor was created inside the function or not:
             */
            boolean resultIsInputGradient = Arrays.stream(tensors).anyMatch( t -> t.getGradient() == result );
            boolean resultIsInputMember = Arrays.stream(tensors).anyMatch( t -> t == result );

            if ( resultIsInputMember || resultIsInputGradient ) {
                int positionInInput;
                if ( resultIsInputGradient )
                    positionInInput = IntStream.range(0, inputs.length).filter(i -> inputs[i].getGradient() == result ).findFirst().getAsInt();
                else
                    positionInInput = IntStream.range(0, inputs.length).filter( i -> inputs[i] == result ).findFirst().getAsInt();

                boolean resultWasIntermediate = ( resultIsInputGradient ? gradIntermediates[positionInInput] : areIntermediates[positionInInput] );

                _wronglyIntermediate = result.isIntermediate() && !resultWasIntermediate;
                _wronglyNonIntermediate = false;
            } else if ( !result.isIntermediate() ) {
                _wronglyIntermediate = false;
                _wronglyNonIntermediate = true;
            } else {
                _wronglyIntermediate = false;
                _wronglyNonIntermediate = false;
            }
        } else {
            _wronglyIntermediate = false;
            _wronglyNonIntermediate = false;
        }
        /*
            Last but not least we return the result
        */
        _result = result;
    }

    /**
     * @return Is {@code true} if the result tensor is wrongfully flagged as intermediate (see {@link Tsr#isIntermediate()}).
     */
    public boolean isWronglyIntermediate() {
        return _wronglyIntermediate;
    }

    /**
     * @return Is {@code true} if the result tensor is wrongfully flagged as non-intermediate (see {@link Tsr#isIntermediate()}).
     */
    public boolean isWronglyNonIntermediate() {
        return _wronglyNonIntermediate;
    }

    /**
     * @return The result tensor returned by the {@link Supplier} lambda passed to this {@link MemValidator}.
     */
    public Tsr<?> getResult() {
        return _result;
    }

}
