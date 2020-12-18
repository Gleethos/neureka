package neureka.autograd;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.calculus.frontend.assembly.FunctionBuilder;

@Accessors( prefix = {"_"} )
@ToString
public class PendingError
{
    @Getter
    private int _toBeReceived;
    @Getter
    private final Tsr<?> _accumulatedError;

    public PendingError( Tsr<?> error, int toBeReceived ) {
        _toBeReceived = toBeReceived;
        _accumulatedError = error;
    }

    public void accumulate( Tsr<?> error ) {
        FunctionBuilder.build(
                "I[ 0 ]<-(I[ 0 ]+I[ 1 ])", false
        ).call( new Tsr[]{ _accumulatedError, error } );
        _toBeReceived--;
    }

    public boolean isFullyAccumulated() {
        return _toBeReceived == 0;
    }

}
