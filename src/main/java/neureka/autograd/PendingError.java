package neureka.autograd;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.assembly.FunctionBuilder;

@Accessors( prefix = {"_"} )
@ToString
public final class PendingError<ValType>
{
    @Getter
    private int _toBeReceived;
    @Getter
    private final Tsr<ValType> _accumulatedError;

    public PendingError( Tsr<ValType> error, int toBeReceived ) {
        _toBeReceived = toBeReceived;
        _accumulatedError = error;
    }

    public void accumulate( Tsr<?> error ) {
        new FunctionBuilder(Neureka.instance().context()).build(
                "I[ 0 ]<-(I[ 0 ]+I[ 1 ])", false
        ).call( new Tsr[]{ _accumulatedError, error } );
        _toBeReceived--;
    }

    public boolean isFullyAccumulated() {
        return _toBeReceived == 0;
    }

}
