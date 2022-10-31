package neureka.backend.main.memory;

import neureka.Tsr;
import neureka.backend.api.Result;

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

    private final Result _result;
    private final boolean _wronglyIntermediate;
    private final boolean _wronglyNonIntermediate;

    /**
     * @param inputs The inputs used by the {@link Supplier} implementation to provide a result.
     * @param resultProvider The callback providing the result which ought to be validated.
     * @return The {@link MemValidator} which ought to validate the provided result.
     */
    public static MemValidator forInputs( Tsr<?>[] inputs, Supplier<Result> resultProvider ) {
        return new MemValidator( inputs, resultProvider );
    }

    private MemValidator( Tsr<?>[] tensors, Supplier<Result> execution ) {
        /*
            Now before calling the function we will do a snapshot of the inputs
            in order to later on verify the output validity with respect
            to the 'intermediate' flag.
        */
        Tsr<?>[] inputs = tensors.clone();
        Boolean[] areIntermediates = Arrays.stream(tensors).map(Tsr::isIntermediate).toArray(Boolean[]::new);
        Boolean[] gradIntermediates = Arrays.stream(tensors).map(t -> (t.hasGradient() && t.gradient().get().isIntermediate())).toArray(Boolean[]::new);
        /*
            Finally, we dispatch the call to the function implementation to get as result!
        */
        Result result = execution.get();
        /*
            Now on to validation!
            First we check if the function executed successfully:
        */
        if ( result == null )
            throw new IllegalStateException( "Failed to execute function! Returned result was null." );
        if ( result.get() == null )
            throw new IllegalStateException( "Failed to execute function! Returned result was null." );

        /*
            After that we analyse the validity of the result
            with respect to memory safety!
            We expect internally created tensors to be flagged as 'intermediate':
            First we check if the result tensor was created inside the function or not:
         */
        boolean resultIsInputGradient = Arrays.stream( tensors ).anyMatch( t -> t.gradient().orElse(null) == result.get() );
        boolean resultIsInputMember   = Arrays.stream( tensors ).anyMatch( t -> t == result.get() );
        /*
            Then we check if this is valid with respect to the "isIntermediate" flag:
         */
        if ( resultIsInputMember || resultIsInputGradient ) {
            int positionInInput =
                    resultIsInputGradient
                        ? IntStream.range( 0, inputs.length )
                                   .filter( i -> inputs[i].gradient().orElse(null) == result.get())
                                   .findFirst()
                                   .getAsInt()
                        : IntStream.range( 0, inputs.length )
                                   .filter( i -> inputs[i] == result.get())
                                   .findFirst()
                                   .getAsInt();

            boolean resultWasIntermediate =
                            resultIsInputGradient
                                ? gradIntermediates[positionInInput]
                                : areIntermediates[positionInInput];

            _wronglyIntermediate = result.get().isIntermediate() && !resultWasIntermediate;
            _wronglyNonIntermediate = false;
        } else if ( !result.get().isIntermediate() ) {
            _wronglyIntermediate = false;
            _wronglyNonIntermediate = true;
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
    public boolean isWronglyIntermediate() { return _wronglyIntermediate; }

    /**
     * @return Is {@code true} if the result tensor is wrongfully flagged as non-intermediate (see {@link Tsr#isIntermediate()}).
     */
    public boolean isWronglyNonIntermediate() { return _wronglyNonIntermediate; }

    /**
     * @return The result tensor returned by the {@link Supplier} lambda passed to this {@link MemValidator}.
     */
    public Result getResult() { return _result; }

}
