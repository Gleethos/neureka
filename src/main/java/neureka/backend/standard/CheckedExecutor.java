package neureka.backend.standard;

import neureka.Tsr;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class CheckedExecutor {

    private final Tsr<?> _result;
    private final boolean _wronglyIntermediate;
    private final boolean _wronglyNonIntermediate;

    public static CheckedExecutor forInputs(Tsr<?>[] inputs, Supplier<Tsr<?>> execution ) {
        return new CheckedExecutor( inputs, execution );
    }

    private CheckedExecutor(Tsr<?>[] tensors, Supplier<Tsr<?>> execution ) {
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

                if ( result.isIntermediate() && !resultWasIntermediate ) {
                    _wronglyIntermediate = true;
                    _wronglyNonIntermediate = false;
                } else {
                    _wronglyIntermediate = false;
                    _wronglyNonIntermediate = false;
                }
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

    public boolean isWronglyIntermediate() {
        return _wronglyIntermediate;
    }

    public boolean isWronglyNonIntermediate() {
        return _wronglyNonIntermediate;
    }

    public Tsr<?> getResult() {
        return _result;
    }

}
