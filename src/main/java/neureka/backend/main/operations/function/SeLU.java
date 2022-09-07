package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

/**
 * The Scaled Exponential Linear Unit, or SELU, is an activation
 * functions that induce self-normalizing properties.
 * The SELU activation function is implemented as:
 * <i>{@code
 *      if      ( x >  0 ) return SCALE * x;
 *      else if ( x <= 0 ) return SCALE * ALPHA * (Math.exp(x) - 1);
 *      else               return Float.NaN;
 * }</i><br>
 * ...where {@code ALPHA == 1.6733} and {@code SCALE == 1.0507}.
 */
public class SeLU extends AbstractActivationOperation
{
    public SeLU() {
        super(ActivationFun.SELU);
    }
}
